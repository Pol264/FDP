#!/usr/bin/env python
import torch
import esm
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
import glob
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from esm2_model import ESM2, ProteinDataset
import gzip

import gzip

class ProteinDataset(Dataset):
    def __init__(self, fasta_file, max_length=1024):
        self.sequences = []
        open_func = gzip.open if fasta_file.endswith(".gz") else open
        with open_func(fasta_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if len(record.seq) <= max_length:
                    self.sequences.append(str(record.seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def compute_perplexity(model, loader, batch_converter, alphabet, device):
    total_loss = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for sequences in loader:
            labels, strs, tokens = batch_converter([(f"seq{i}", seq) for i, seq in enumerate(sequences)])
            tokens = tokens.to(device)
            logits = model(tokens)['logits']
            loss = cross_entropy(logits.transpose(1, 2), tokens, ignore_index=alphabet.padding_idx, reduction='sum')
            total_loss += loss.item()
            total_tokens += (tokens != alphabet.padding_idx).sum().item()
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


def main():
    fabric = Fabric(accelerator="cuda", devices=3, strategy=DDPStrategy(), precision="bf16-mixed")
    fabric.launch()
    
    This part is for OUR MODELS
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()
    model = ESM2(num_layers=6, embed_dim=320, attention_heads=20, alphabet=alphabet)
    checkpoint = torch.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/esm2_t6_8M_UR50D.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.half().to(fabric.device)
    #model = esm.pretrained.esm2_t6_8M_UR50D().half().to(fabric.device)
    results = {}
    val_dataset_path = "./uniref50_dataset/uniref50_validation_sequences.fasta.gz"
    val_dataset = ProteinDataset(val_dataset_path)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=8, shuffle=False)
    val_loader = fabric.setup_dataloaders(val_loader)
    val_perplexity = compute_perplexity(model, val_loader, batch_converter, alphabet, fabric.device)
    results["validation"] = val_perplexity
    print(f"Validation Dataset Perplexity: {val_perplexity}", flush=True)

    datasets = glob.glob("./mgnify_datasets/10000_sequences_*_dataset_rep_seq_sorted_by_length.fasta")
    

    for dataset_path in datasets:
        dataset = ProteinDataset(dataset_path)
        loader = DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)
        loader = fabric.setup_dataloaders(loader)

        perplexity = compute_perplexity(model, loader, batch_converter, alphabet, fabric.device)
        results[dataset_path] = perplexity
        print(f"Dataset: {dataset_path}, Perplexity: {perplexity}",flush=True)
    print("35M_baseline",flush=True)
if __name__ == "__main__":
    main()
