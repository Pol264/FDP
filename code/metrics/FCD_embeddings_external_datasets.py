import torch
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESMEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim=1280, latent_dim=64):
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

def load_pretrained_autoencoder(weights_path="autoencoder_optimized_650M_baseline.pth"):
    model = ESMEmbeddingAutoencoder().to(device)
    state_dict = torch.load(weights_path)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.module.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    return model

def calculate_fcd(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(cov1 + cov2 - 2 * covmean)

def compute_mean(memmap_array, chunk_size=100000):
    total = np.zeros(memmap_array.shape[1], dtype=np.float64)
    n = 0
    for i in range(0, memmap_array.shape[0], chunk_size):
        chunk = memmap_array[i:i+chunk_size]
        total += np.sum(chunk, axis=0)
        n += chunk.shape[0]
    return total / n

def compute_covariance(memmap_array, mean, chunk_size=100000):
    d = memmap_array.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)
    n = 0
    for i in range(0, memmap_array.shape[0], chunk_size):
        chunk = memmap_array[i:i+chunk_size]
        diff = chunk - mean
        cov += diff.T @ diff
        n += chunk.shape[0]
    return cov / (n - 1)





def compute_unfamiliarity(model, embeddings):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructions = model(embeddings_tensor)
        mse_loss = torch.mean((reconstructions - embeddings_tensor)**2, dim=1)
        unfamiliarity = torch.log(mse_loss).cpu().numpy()
    return unfamiliarity




print("loading embeddings train",flush=True)
embeddings_train = np.load("/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/650M_embedding_all_all.npy", mmap_mode='r')
print("loading_embeddigs_train_finished",flush=True)
embeddings_secondary_structure = np.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/embeddings_secondary_structure_650M.npz")['embeddings']
print("loading_embeddings_secondary_structure_finished",flush=True)
mu_train = compute_mean(embeddings_train)
cov_train = compute_covariance(embeddings_train, mu_train)
cov_inv_train = np.linalg.inv(cov_train)
print("loading_autoencoder",flush=True)
autoencoder = load_pretrained_autoencoder()
print("autoencoder_loaded",flush=True)
# Validation metrics
mu_secondary_structure, cov_secondary_structure = np.mean(embeddings_secondary_structure, axis=0), np.cov(embeddings_secondary_structure, rowvar=False)
print(f"secondary_structure FCD: {calculate_fcd(mu_train, cov_train, mu_secondary_structure, cov_secondary_structure):.4f}",flush=True)
cosine_sim = cosine_similarity([np.mean(embeddings_train, axis=0)],[np.mean(embeddings_secondary_structure, axis=0)])[0][0]
print(f"secondary_structure Cosine Similarity: {cosine_sim:.4f}",flush=True)

#print(f"Validation Cosine Similarity: {np.mean(cosine_similarity(embeddings_train, embeddings_val)):.4f}")
print(f"secondary_structure Mahalanobis: {np.mean([distance.mahalanobis(v, mu_train, cov_inv_train) for v in embeddings_secondary_structure]):.4f}")
print(f"secondary_structure Unfamiliarity: {np.mean(compute_unfamiliarity(autoencoder, embeddings_secondary_structure)):.4f}")
#perplexity_val = compute_perplexity(embeddings_val, mu_train, cov_train)
#print(f"Validation Perplexity: {perplexity_val:.4f}",flush=True)


embeddings_remote_homology = np.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/datasets_prove_metrics/embeddings_remote_homology_650M.npz")['embeddings']
print("loading_embeddings_remote_homology_finished",flush=True)
mu_train = compute_mean(embeddings_train)
cov_train = compute_covariance(embeddings_train, mu_train)
cov_inv_train = np.linalg.inv(cov_train)
print("loading_autoencoder",flush=True)
autoencoder = load_pretrained_autoencoder()
print("autoencoder_loaded",flush=True)
# Validation metrics
mu_remote_homology, cov_remote_homology = np.mean(embeddings_remote_homology, axis=0), np.cov(embeddings_remote_homology, rowvar=False)
print(f"remote_homology FCD: {calculate_fcd(mu_train, cov_train, mu_remote_homology, cov_remote_homology):.4f}",flush=True)
cosine_sim_remote_homology = cosine_similarity([np.mean(embeddings_train, axis=0)],[np.mean(embeddings_remote_homology, axis=0)])[0][0]
print(f"remote_homology Cosine Similarity: {cosine_sim_remote_homology:.4f}",flush=True)

print(f"remote_homology Mahalanobis: {np.mean([distance.mahalanobis(v, mu_train, cov_inv_train) for v in embeddings_remote_homology]):.4f}")
print(f"remote_homology Unfamiliarity: {np.mean(compute_unfamiliarity(autoencoder, embeddings_remote_homology)):.4f}")
print("650M")
#perplexity_val = compute_perplexity(embeddings_val, mu_train, cov_train)
#print(f"Validation Perplexity: {perplexity_val:.4f}",flush=True)


