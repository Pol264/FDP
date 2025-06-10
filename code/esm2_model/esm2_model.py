#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List, Tuple, Generator
import argparse
import gzip
import torch
import lightning
import esm
import torch.nn as nn
import torch.nn.functional as F
from esm.data import Alphabet
import typing as T
from lightning.fabric import Fabric
from pathlib import Path
from Bio import SeqIO
from lightning.fabric.strategies import DDPStrategy
import wandb
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import os
import torch
from itertools import islice
from torch.utils.data import RandomSampler
import glob

os.environ["WANDB_MODE"] = "offline"
wandb.init(
        project="ESM-2 model"
             )
##############################################
# DataLoader
##############################################
class ProteinDataset(Dataset):
    def __init__(self, fasta_file, max_sequences=None, max_length=512):
        self.sequences = self._load_sequences(fasta_file, max_sequences, max_length)

    def _load_sequences(self, fasta_file, max_sequences, max_length):
        sequences = []
        count = 0
        handle = gzip.open(fasta_file, "rt") if fasta_file.endswith(".gz") else open(fasta_file, "rt")
        for record in SeqIO.parse(handle, "fasta"):
            if len(record.seq) <= max_length:
                sequences.append((record.id, str(record.seq)))
                count += 1
                if max_sequences and count >= max_sequences:
                    break
        handle.close()
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        header, seq = self.sequences[idx]
        return header, seq


##############################################
# ESM2 Model Definition
##############################################
class ESM2(nn.Module):
    def __init__(
            self,
            num_layers: int = 33,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[Alphabet, str] = "ESM-1b",
            token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        from esm.modules import TransformerLayer, ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / torch.clamp(src_lengths, min=1)
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        x = x.transpose(0, 1)  # (B, T, E) -> (T, B, E)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) -> (B, T, E)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result


##############################################
# Main Training
##############################################
from tqdm import tqdm


def train_model(args, model, dataset, fabric, optimizer, loss_fn, batch_converter, alphabet, start_step=0):
    import time
    from itertools import islice

    def mask_tokens(tokens, alphabet, mask_ratio=0.15):
        probability_matrix = torch.full(tokens.shape, mask_ratio)
        special_tokens_mask = (tokens == alphabet.padding_idx) | (tokens == alphabet.cls_idx) | (tokens == alphabet.eos_idx)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        tokens[masked_indices] = alphabet.mask_idx
        return tokens, masked_indices

    print("Starting training...")
    global_step = start_step
    checkpoint_interval = 100000
    accumulation_steps = 2

    num_workers = min(8, multiprocessing.cpu_count() // 2) #THE NUMBER OF WORKERS MUST MATCH WITH NUMBER OF CPUS PER TASK
    steps_per_epoch = len(dataset) // args.max_tokens_per_batch
    start_epoch = start_step // steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0

        sampler = RandomSampler(
            dataset,
            replacement=False,
            generator=torch.Generator().manual_seed(42 + epoch)  # epoch-dependent seed for different shuffling
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.max_tokens_per_batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        dataloader = fabric.setup_dataloaders(dataloader)

        current_epoch_index = global_step // steps_per_epoch
        steps_into_current_epoch = global_step % steps_per_epoch

        if epoch == current_epoch_index and steps_into_current_epoch > 0:
            dataloader = islice(dataloader, steps_into_current_epoch, None)

        for step, (headers, seqs) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", initial=steps_into_current_epoch),
            start=steps_into_current_epoch + 1
        ):
            global_step += 1

            batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(headers, seqs)))
            batch_labels_tensor = batch_tokens.clone()

            batch_tokens, masked_indices = mask_tokens(batch_tokens, alphabet)
            batch_tokens = fabric.to_device(batch_tokens)

            outputs = model(batch_tokens)
            logits = outputs["logits"]

            masked_logits = logits[masked_indices]
            masked_labels = batch_labels_tensor[masked_indices].to(fabric.device)

            if masked_logits.numel() == 0:
                continue

            loss = loss_fn(masked_logits, masked_labels)
            fabric.backward(loss)

            wandb.log({
                "step": global_step,
                "loss": loss
            })

            if torch.isnan(loss) or torch.isinf(loss):
                break

            if global_step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += loss.item()
            batch_count += 1

            if global_step % checkpoint_interval == 0:
                avg_loss = epoch_loss / batch_count
                print(f"Checkpoint Step {global_step}, Avg Loss: {avg_loss:.4f}", flush=True)
                checkpoint_dir = os.path.dirname(args.checkpoint_path)
                checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pth")
                fabric.save(checkpoint_filename, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})

        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')
        print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f} seconds", flush=True)

        base_name = os.path.splitext(os.path.basename(args.model_path))[0]
        epoch_model_path = os.path.join(
            os.path.dirname(args.model_path),
            f"{base_name}_epoch_{epoch + 1}.pt"
        )
        fabric.save(epoch_model_path, model.state_dict())
        print(f"Model checkpoint saved at {epoch_model_path}", flush=True)

    fabric.save(args.model_path, model.state_dict())
    print(f"Model saved at {args.model_path}", flush=True)

def load_latest_checkpoint_with_fallback(checkpoint_dir, device):
    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth")),
        key=os.path.getmtime, reverse=True
    )
    for checkpoint_path in checkpoint_files[:2]:  # Try latest, then penultimate
        try:
            print(f"Attempting to load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            step = int(Path(checkpoint_path).stem.split("_")[-1])
            print(f"Successfully loaded checkpoint at step {step}")
            return checkpoint, step
        except Exception as e:
            print(f"Failed to load {checkpoint_path}: {e}")
    raise RuntimeError("No valid checkpoint could be loaded!")

##############################################
# Main Function
##############################################
def main():
    parser = argparse.ArgumentParser(description="Train ESM2 model on a FASTA file (UniRef90)")
    parser.add_argument("--fasta", type=str, required=True, help="Path to the FASTA file (can be gzipped)")
    parser.add_argument("--max_tokens_per_batch", type=int, default=8, help="Max tokens per batch")
    parser.add_argument("--num_layers", type=int, help="Number of Transformer layers")
    parser.add_argument("--embed_dim", type=int, help="Embedding dimension")
    parser.add_argument("--attention_heads", type=int, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of sequences to load")
    parser.add_argument("--model_path", type=str, default="esm2.pt", help="Path to save the model")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to save the checkpoint")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume training")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    alphabet = Alphabet.from_architecture("ESM-1b")
    batch_converter = alphabet.get_batch_converter()

    strategy = DDPStrategy(find_unused_parameters=True)
    fabric = Fabric(accelerator="cuda", devices=3, strategy=strategy, precision="bf16-mixed")
    fabric.launch()

    dataset = ProteinDataset(args.fasta, args.max_sequences)
    

    start_step = 0  # Defined here for proper use below
    '''
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=fabric.device)
        checkpoint_filename = Path(args.resume_from_checkpoint).stem
        start_step = int(checkpoint_filename.split("_")[-1])
        model_state, optim_state = checkpoint["model"], checkpoint["optimizer"]
        print(f"Resuming training from checkpoint at step {start_step}")
    else:
        model_state, optim_state = None, None
        print("Starting training from scratch.")
    '''
    if args.resume_from_checkpoint:
        checkpoint_dir = os.path.dirname(args.resume_from_checkpoint)
        try:
            checkpoint, start_step = load_latest_checkpoint_with_fallback(checkpoint_dir, fabric.device)
            model_state, optim_state = checkpoint["model"], checkpoint["optimizer"]
            print(f"Resuming training from checkpoint at step {start_step}")
        except Exception as e:
            print(f"Error loading checkpoints: {e}")
            model_state, optim_state, start_step = None, None, 0
            print("Starting training from scratch.")
    else:
        model_state, optim_state, start_step = None, None, 0
        print("Starting training from scratch.")
        
    
    model = ESM2(
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        attention_heads=args.attention_heads,
        alphabet=alphabet,
        token_dropout=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"Total number of parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    fabric.print("Before fabric.setup()")

    model, optimizer = fabric.setup(model, optimizer)
    fabric.print("After fabric.setup()")

    if model_state:
        model.load_state_dict(model_state)
    if optim_state:
        optimizer.load_state_dict(optim_state)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=alphabet.padding_idx)

    train_model(args, model, dataset, fabric, optimizer, loss_fn, batch_converter, alphabet, start_step)

    '''
    import glob

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth")), key=os.path.getmtime, reverse=True)

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]  # Select the most recent checkpoint
        print(f"Loading latest checkpoint: {latest_checkpoint}...")
        checkpoint = torch.load(latest_checkpoint, map_location=fabric.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from checkpoint: {latest_checkpoint}")
    
    try:
        model.load_state_dict(torch.load("./models/8M_validation_prova.pt", map_location=fabric.device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    '''

if __name__ == "__main__":
    main()
