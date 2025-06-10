import torch
import esm
import gzip
from Bio import SeqIO
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("/leonardo_work/EUHPC_A05_043/TFG_pjardi")


from esm2_model import ESM2
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Load model

esm_path = "/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/esm2_t33_650M_UR50D.pt"
model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_path)
batch_converter = alphabet.get_batch_converter()

esm_model = model.half().to(device).eval()

# Load sequences
def load_sequences_from_fasta_gz(filepath, max_length=1024):
    with gzip.open(filepath, 'rt') as file:
        return [str(record.seq) for record in SeqIO.parse(file, 'fasta') if len(record.seq) <= max_length]

def load_sequences_from_fasta(filepath, max_length=1024):
    with open(filepath, 'rt') as file:
        return [str(record.seq) for record in SeqIO.parse(file, 'fasta') if len(record.seq) <= max_length]

# Tokenization (parallel)
def tokenize_batch(data):
    _, _, batch_tokens = batch_converter(data)
    return batch_tokens

# Embedding function
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

# Run embedding generation
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = {
        "remote_homology": "/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/primary_sequences_remote_homology_train.fasta",
        "secondary_structure": "/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/primary_sequences_secondary_structure_train.fasta"
    }

    repr_layer = 33
    for name, path in datasets.items():
        print(f"\nProcessing {name}...", flush=True)
        start = time.time()
        seqs = load_sequences_from_fasta_gz(path) if path.endswith(".gz") else load_sequences_from_fasta(path)
        embeddings = compute_embeddings(seqs, repr_layer, batch_size=16, num_workers=8)
        np.savez_compressed(f"/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/embeddings_{name}_650M.npz", embeddings=embeddings)
        print(f"Done {name} in {time.time() - start:.2f}s", flush=True)

    print("\nAll done.", flush=True)

