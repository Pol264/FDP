import torch
import esm
import gzip
from Bio import SeqIO
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
from esm2_model import ESM2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

alphabet = esm.Alphabet.from_architecture("ESM-1b")
batch_converter = alphabet.get_batch_converter()
esm_model = ESM2(num_layers=6, embed_dim=320, attention_heads=20, alphabet=alphabet)
state_dict = torch.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/8M_model.pth", map_location="cpu")
esm_model.load_state_dict(state_dict)
esm_model.to(device).half().eval()

def load_sequences_from_fasta_gz(filepath, max_length=512):
    with gzip.open(filepath, 'rt') as file:
        return [str(record.seq) for record in SeqIO.parse(file, 'fasta') if len(record.seq) <= max_length]

def load_sequences_from_fasta(filepath, max_length=512):
    with open(filepath, 'rt') as file:
        return [str(record.seq) for record in SeqIO.parse(file, 'fasta') if len(record.seq) <= max_length]

def tokenize_batch(data):
    _, _, batch_tokens = batch_converter(data)
    return batch_tokens

def compute_embeddings(sequences, repr_layer, batch_size=16, num_workers=8):
    results = []
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_tokens = executor.submit(tokenize_batch, batch_data).result()
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                output = esm_model(batch_tokens, repr_layers=[repr_layer])
                token_reps = output["representations"][repr_layer]
                for j, tokens in enumerate(batch_tokens):
                    seq_len = (tokens != alphabet.padding_idx).sum().item()
                    emb = token_reps[j, 1:seq_len - 1].mean(0).float().cpu().numpy().astype(np.float16)
                    results.append(emb)
    return np.vstack(results)

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = {
        "validation": "./uniref50_dataset/uniref50_validation_sequences.fasta.gz",
        "test_1_40_60": "./mgnify_dataset/10000_sequences_folder/10000_sequences_40_60_cluster_filtered_by_512_0_9_threshold_1_dataset_rep_seq_sorted_by_length.fasta",
        "test_1_60_80": "./mgnify_dataset/10000_sequences_folder/10000_sequences_60_80_cluster_filtered_by_512_0_9_threshold_1_dataset_rep_seq_sorted_by_length.fasta",
        "test_1_80_90": "./mgnify_dataset/10000_sequences_folder/10000_sequences_80_90_cluster_filtered_by_512_0_9_threshold_1_dataset_rep_seq_sorted_by_length.fasta",
        "test_1_90_100": "./mgnify_dataset/10000_sequences_folder/10000_sequences_90_100_cluster_filtered_by_512_0_9_threshold_1_dataset_rep_seq_sorted_by_length.fasta",
        "test_2_40_60": "./mgnify_dataset/10000_sequences_folder/10000_sequences_40_60_cluster_filtered_by_512_0_9_threshold_2_dataset_rep_seq_sorted_by_length.fasta",
        "test_2_60_80": "./mgnify_dataset/10000_sequences_folder/10000_sequences_60_80_cluster_filtered_by_512_0_9_threshold_2_dataset_rep_seq_sorted_by_length.fasta",
        "test_2_80_90": "./mgnify_dataset/10000_sequences_folder/10000_sequences_80_90_cluster_filtered_by_512_0_9_threshold_2_dataset_rep_seq_sorted_by_length.fasta",
        "test_2_90_100": "./mgnify_dataset/10000_sequences_folder/10000_sequences_90_100_cluster_filtered_by_512_0_9_threshold_2_dataset_rep_seq_sorted_by_length.fasta",
        "test_3_40_60": "./mgnify_dataset/10000_sequences_folder/10000_sequences_40_60_cluster_filtered_by_512_0_9_threshold_3_dataset_rep_seq_sorted_by_length.fasta",
        "test_3_60_80": "./mgnify_dataset/10000_sequences_folder/10000_sequences_60_80_cluster_filtered_by_512_0_9_threshold_3_dataset_rep_seq_sorted_by_length.fasta",
        "test_3_80_90": "./mgnify_dataset/10000_sequences_folder/10000_sequences_80_90_cluster_filtered_by_512_0_9_threshold_3_dataset_rep_seq_sorted_by_length.fasta",
        "test_3_90_100": "./mgnify_dataset/10000_sequences_folder/10000_sequences_90_100_cluster_filtered_by_512_0_9_threshold_3_dataset_rep_seq_sorted_by_length.fasta"
    }

    repr_layer = 6
    for name, path in datasets.items():
        print(f"\nProcessing {name}...", flush=True)
        start = time.time()
        seqs = load_sequences_from_fasta_gz(path) if path.endswith(".gz") else load_sequences_from_fasta(path)
        embeddings = compute_embeddings(seqs, repr_layer, batch_size=16, num_workers=8)
        np.savez_compressed(f"/leonardo_work/EUHPC_A05_043/TFG_pjardi/uniref50_dataset/embeddings_8M_our_model_validation_and_testing/embeddings_{name}.npz", embeddings=embeddings)
        print(f"Done {name} in {time.time() - start:.2f}s", flush=True)

    print("\nAll done.", flush=True)
