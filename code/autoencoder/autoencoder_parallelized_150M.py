import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESMEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim=320, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class NpyDataset(Dataset):
    def __init__(self, path):
        print(f"Loading dataset from: {path}")
        self.data = np.load(path, mmap_mode='r')
        print(f"Dataset shape: {self.data.shape}")

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

def train_autoencoder(model, embedding_path, epochs=1, batch_size=4096, lr=1e-3):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    dataset = NpyDataset(embedding_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True, persistent_workers=True
    )

    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss, count = 0.0, 0
        print(f"\nStarting epoch {epoch+1}/{epochs}...")

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                recon = model(batch)
                loss = loss_fn(recon, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch.size(0)
            count += batch.size(0)

            if (batch_idx + 1) % 50 == 0:
                percent = 100.0 * count / len(dataset)
                print(f"  [{count}/{len(dataset)}] {percent:.2f}% done | Batch Loss: {loss.item():.6f}", flush=True)

        avg_loss = total_loss / count
        duration = time.time() - start_time
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.6f} - Time: {duration/60:.2f} min", flush=True)

    torch.save(model.state_dict(), "autoencoder_optimized_150M_our_model.pth")
    print("\nModel saved as autoencoder_optimized_150M_our_model.pth",flush=True)

if __name__ == "__main__":
    embeddings_training = "/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/150M_our_model_embedding_all_all.npy"
    print("Initializing model...", flush=True)
    model = ESMEmbeddingAutoencoder(input_dim=640, latent_dim=64)
    train_autoencoder(model, embeddings_training)

