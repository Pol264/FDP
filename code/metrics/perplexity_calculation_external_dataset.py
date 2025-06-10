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
    fabric = Fabric(accelerator="cuda", devices=1, strategy=DDPStrategy(), precision="bf16-mixed")
    fabric.launch()
    '''
    This part is for OUR MODELS
    alphabet = esm.Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()
    model = ESM2(num_layers=6, embed_dim=320, attention_heads=20, alphabet=alphabet)
    checkpoint = torch.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/esm2_t6_8M_UR50D.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    '''
    model, alphabet = esm.pretrained.load_model_and_alphabet_local("/leonardo_work/EUHPC_A05_043/TFG_pjardi/models/esm2_t33_650M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()
    model = model.half().to(fabric.device)
    #model = esm.pretrained.esm2_t6_8M_UR50D().half().to(fabric.device)
    results = {}
    secondary_structure_dataset_path = "/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/primary_sequences_secondary_structure_train.fasta"
    secondary_structure_dataset = ProteinDataset(secondary_structure_dataset_path)
    secondary_structure_loader = DataLoader(secondary_structure_dataset, batch_size=16, num_workers=8, shuffle=False)
    secondary_structure_loader = fabric.setup_dataloaders(secondary_structure_loader)
    secondary_structure_perplexity = compute_perplexity(model, secondary_structure_loader, batch_converter, alphabet, fabric.device)
    results["secondary_structure"] = secondary_structure_perplexity
    print(f"secondary_structure Dataset Perplexity: {secondary_structure_perplexity}", flush=True)
    
    remote_homology_dataset_path = "/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/primary_sequences_remote_homology_train.fasta"
    remote_homology_dataset = ProteinDataset(remote_homology_dataset_path)
    remote_homology_loader = DataLoader(remote_homology_dataset, batch_size=16, num_workers=8, shuffle=False)
    remote_homology_loader = fabric.setup_dataloaders(remote_homology_loader)
    remote_homology_perplexity = compute_perplexity(model, remote_homology_loader, batch_converter, alphabet, fabric.device)
    results["remote_homology"] = remote_homology_perplexity
    print(f"remote_homology Dataset Perplexity: {remote_homology_perplexity}", flush=True)


    

if __name__ == "__main__":
  main()
