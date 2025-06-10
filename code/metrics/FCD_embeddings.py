import torch
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESMEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim=640, latent_dim=64):
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

def load_pretrained_autoencoder(weights_path="autoencoder_optimized_150M_our_model.pth"):
    model = ESMEmbeddingAutoencoder().to(device)
    state_dict = torch.load(weights_path)

    # Handle 'DataParallel' saved state_dict keys
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
#embeddings_train = np.load("/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/8M_embedding_all_all.npy")
embeddings_train = np.load("/leonardo_scratch/large/userexternal/pjardiya/jobs_embeddings/all_combined/150M_our_model_embedding_all_all.npy", mmap_mode='r')
print("loading_embeddigs_train_finished",flush=True)
embeddings_val = np.load("/leonardo_work/EUHPC_A05_043/TFG_pjardi/uniref50_dataset/embeddings_150M_our_model_validation_and_testing/embeddings_validation.npz")['embeddings']
print("loading_embeddings_validation_finished",flush=True)
mu_train = compute_mean(embeddings_train)
cov_train = compute_covariance(embeddings_train, mu_train)
cov_inv_train = np.linalg.inv(cov_train)
print("loading_autoencoder",flush=True)
autoencoder = load_pretrained_autoencoder()
print("autoencoder_loaded",flush=True)
# Validation metrics
mu_val, cov_val = np.mean(embeddings_val, axis=0), np.cov(embeddings_val, rowvar=False)
print(f"Validation FCD: {calculate_fcd(mu_train, cov_train, mu_val, cov_val):.4f}",flush=True)
cosine_sim = cosine_similarity([np.mean(embeddings_train, axis=0)],[np.mean(embeddings_val, axis=0)])[0][0]
print(f"Validation Cosine Similarity: {cosine_sim:.4f}",flush=True)

#print(f"Validation Cosine Similarity: {np.mean(cosine_similarity(embeddings_train, embeddings_val)):.4f}")
print(f"Validation Mahalanobis: {np.mean([distance.mahalanobis(v, mu_train, cov_inv_train) for v in embeddings_val]):.4f}")
print(f"Validation Unfamiliarity: {np.mean(compute_unfamiliarity(autoencoder, embeddings_val)):.4f}")
#perplexity_val = compute_perplexity(embeddings_val, mu_train, cov_train)
#print(f"Validation Perplexity: {perplexity_val:.4f}",flush=True)

# Test loop

import os

test_dir = "/leonardo_work/EUHPC_A05_043/TFG_pjardi/uniref50_dataset/embeddings_150M_our_model_validation_and_testing/"
filenames = [
    "embeddings_test_1_90_100.npz", "embeddings_test_1_80_90.npz", "embeddings_test_1_60_80.npz",
    "embeddings_test_1_40_60.npz", "embeddings_test_2_90_100.npz", "embeddings_test_2_80_90.npz",
    "embeddings_test_2_60_80.npz", "embeddings_test_2_40_60.npz", "embeddings_test_3_90_100.npz",
    "embeddings_test_3_80_90.npz", "embeddings_test_3_60_80.npz", "embeddings_test_3_40_60.npz"
]
test_files = [os.path.join(test_dir, f) for f in filenames]

for fname in test_files:
    test_embeddings = np.load(fname)['embeddings']
    mu_test = np.mean(test_embeddings, axis=0)
    cov_test = np.cov(test_embeddings, rowvar=False)

    fcd = calculate_fcd(mu_train, cov_train, mu_test, cov_test)
    cosine_sim = cosine_similarity([np.mean(embeddings_train, axis=0)],[np.mean(test_embeddings, axis=0)])[0][0]

    #cosine_sim = np.mean(cosine_similarity(embeddings_train, test_embeddings))
    mahal = np.mean([distance.mahalanobis(v, mu_train, cov_inv_train) for v in test_embeddings])
    unfam = np.mean(compute_unfamiliarity(autoencoder, test_embeddings))
    #perplexity_test = compute_perplexity(test_embeddings, mu_train, cov_train)
    perplexity_test=0
    print(f"{fname}: FCD={fcd:.4f}, Cosine={cosine_sim:.4f}, Mahalanobis={mahal:.4f}, Perplexity={perplexity_test:.4f}, Unfamiliarity={unfam:.4f}",flush=True)

