import os
import torch
import esm
import gzip
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import time
import torch.multiprocessing as mp
import argparse

class FastaDataset(Dataset):
    def __init__(self, gz_path, max_len=512, start=0, end=None):
        self.seqs = []
        with gzip.open(gz_path, 'rt') as f:
            for i, rec in enumerate(SeqIO.parse(f, 'fasta')):
                if len(rec.seq) <= max_len:
                    if i >= start and (end is None or i < end):
                        self.seqs.append((str(rec.seq), i))
                    if end is not None and i >= end:
                        break

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]

def save_embeddings(embeddings, indices, output_dir, job_idx, gpu_rank, batch_id):
    fname = os.path.join(output_dir, f"job{job_idx}_gpu{gpu_rank}_batch{batch_id:05d}.npz")
    np.savez_compressed(fname, **{str(idx): emb for idx, emb in zip(indices, embeddings)})

def worker(rank, world_size, data_path, output_dir, repr_layer, batch_size, start_idx, end_idx, job_idx):
    start_time = time()
    device = torch.device(f"cuda:{rank}")
    esm_path = "/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/model_8M.pth"
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_path)
    model = model.half().to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    ds = FastaDataset(data_path, start=start_idx, end=end_idx)
    total = len(ds)
    local_start = (total * rank) // world_size
    local_end = (total * (rank + 1)) // world_size
    sub_ds = torch.utils.data.Subset(ds, list(range(local_start, local_end)))
    dl = DataLoader(sub_ds, batch_size=batch_size, num_workers=4, shuffle=False)

    print(f"[Job {job_idx}, GPU {rank}] Total sequences assigned: {local_end - local_start}")

    batch_id = 0
    total_processed = 0
    for seq_batch in dl:
        seqs, idxs = seq_batch
        data = [(f"seq_{idx}", seq) for seq, idx in zip(seqs, idxs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(tokens, repr_layers=[repr_layer])
        reps = out["representations"][repr_layer]

        embeddings = []
        for i, tok in enumerate(tokens):
            length = (tok != alphabet.padding_idx).sum().item() - 2
            emb = reps[i, 1:1+length].mean(0).cpu().float().numpy().astype(np.float16)
            embeddings.append(emb)

        embeddings = np.stack(embeddings, 0)

        save_embeddings(embeddings, idxs.tolist(), output_dir, job_idx, rank, batch_id)

        del tokens, out, reps, embeddings
        torch.cuda.empty_cache()

        batch_id += 1
        total_processed += len(seqs)

    duration = time() - start_time
    print(f"[Job {job_idx}, GPU {rank}] Finished. Batches: {batch_id}, Sequences: {total_processed}, Time: {duration / 60:.2f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total_seqs", type=int, required=True)
    parser.add_argument("--job_idx", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, required=True)
    args = parser.parse_args()

    batch_size = 4
    repr_layer = 6
    world_size = 3  # GPUs per job

    os.makedirs(args.output_dir, exist_ok=True)

    seqs_per_job = args.total_seqs // args.n_jobs
    job_start = args.job_idx * seqs_per_job
    job_end = args.total_seqs if args.job_idx == args.n_jobs - 1 else job_start + seqs_per_job

    print(f"[Job {args.job_idx}] Processing sequences from {job_start} to {job_end} "
          f"({job_end - job_start} sequences total)")

    mp.spawn(
        worker,
        args=(world_size, args.data_path, args.output_dir, repr_layer, batch_size, job_start, job_end, args.job_idx),
        nprocs=world_size,
        join=True
    )
