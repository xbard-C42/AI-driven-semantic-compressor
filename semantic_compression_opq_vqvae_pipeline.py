# semantic_compression_opq_vqvae_pipeline.py
"""
Enhanced pipeline combining FAISS OPQ+PQ and a VQ-VAE for semantic compression on Wikitext-2,
with reproducible training, gradient clipping, visual curve, and export of all test artifacts.
Prerequisites:
    pip install datasets sentence-transformers numpy torch faiss-cpu scikit-learn pandas matplotlib
"""
import os
import json
import random
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd

# 1. Configuration
CONFIG = {
    "dataset_split": "train",
    "sample_size": 10000,
    "batch_size": 128,
    "pq_M": 8,
    "pq_bits": 8,
    "vqvae_embedding_dim": 64,
    "vqvae_num_embeddings": 512,
    "vqvae_hidden_dim": 128,
    "vqvae_learning_rate": 2e-4,
    "vqvae_epochs": 10,
    "test_fraction": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_faiss_gpu": False,
    "seed": 42
}

# Smart FAISS import with GPU fallback
try:
    import faiss
    from faiss import StandardGpuResources, index_cpu_to_gpu, index_gpu_to_cpu
    GPU_FAISS_AVAILABLE = CONFIG["use_faiss_gpu"]
    if GPU_FAISS_AVAILABLE:
        print("🚀 GPU-accelerated FAISS functions loaded!")
    else:
        print("💻 Using CPU-only FAISS (fallback).")
except ModuleNotFoundError:
    raise ModuleNotFoundError("FAISS library not found. Please install via: `pip install faiss-cpu`")

# 2. Dataset
class TextDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx]

# 3. VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        flat_z = z.view(-1, self.embedding_dim)
        # Compute distances
        dists = (flat_z.pow(2).sum(1, keepdim=True)
                 - 2 * flat_z @ self.embedding.weight.t()
                 + self.embedding.weight.pow(2).sum(1))
        indices = torch.argmin(dists, dim=1)
        quantized = self.embedding(indices).view_as(z)
        # Losses
        e_loss = torch.mean((quantized.detach() - z).pow(2))
        q_loss = torch.mean((quantized - z.detach()).pow(2))
        loss = q_loss + self.commitment_cost * e_loss
        # Straight-through
        quantized = z + (quantized - z).detach()
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, q_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        recon_loss = torch.mean((x_recon - x).pow(2))
        return x_recon, recon_loss + q_loss

# 4. Pipeline
def main():
    # Reproducibility
    SEED = CONFIG["seed"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("\n🎪 Starting Enhanced Semantic Compression Pipeline with Reconstruction Export 🎪\n")

    # Load & sample text
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=CONFIG["dataset_split"])
    sentences = [s for s in ds["text"] if len(s.split()) > 5]
    if len(sentences) < CONFIG["sample_size"]:
        raise ValueError(f"Not enough sentences ({len(sentences)} < {CONFIG['sample_size']}).")
    corpus = random.sample(sentences, k=CONFIG["sample_size"])
    print(f"✅ Sampled {len(corpus)} sentences.\n")

    # Generate embeddings
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=CONFIG["device"])
    embs = sbert.encode(corpus,
                        batch_size=CONFIG["batch_size"],
                        show_progress_bar=True)
    embs = torch.from_numpy(np.array(embs)).float()
    print(f"✅ Embeddings: {embs.shape}\n")

    # Train/test split
    train_embs, test_embs = train_test_split(
        embs, test_size=CONFIG["test_fraction"], random_state=SEED
    )
    print(f"✅ Train/Test split: {len(train_embs)}/{len(test_embs)}\n")

    # FAISS OPQ+PQ
    print("🔍 Training FAISS OPQ+PQ index...")
    all_np = embs.numpy().astype("float32")
    d = all_np.shape[1]
    cpu_index = faiss.IndexPreTransform(
        faiss.OPQMatrix(d, CONFIG["pq_M"]),
        faiss.IndexPQ(d, CONFIG["pq_M"], CONFIG["pq_bits"])
    )
    cpu_index.train(all_np)
    cpu_index.add(all_np)
    if CONFIG["use_faiss_gpu"] and GPU_FAISS_AVAILABLE:
        res = StandardGpuResources()
        gpu_index = index_cpu_to_gpu(res, 0, cpu_index)
        index = gpu_index
    else:
        print("💻 Using CPU-only FAISS indexing.")
        index = cpu_index
    print("✅ FAISS index ready!\n")

    # Build & train VQ-VAE
    vqvae = VQVAE(d,
                  CONFIG["vqvae_hidden_dim"],
                  CONFIG["vqvae_embedding_dim"],
                  CONFIG["vqvae_num_embeddings"]).to(CONFIG["device"])
    optimizer = optim.Adam(vqvae.parameters(), lr=CONFIG["vqvae_learning_rate"])
    loader = DataLoader(TextDataset(train_embs),
                        batch_size=CONFIG["batch_size"],
                        shuffle=True)

    print(f"🎯 Training VQ-VAE on {CONFIG['device']} for {CONFIG['vqvae_epochs']} epochs...\n")
    epoch_losses = []
    for epoch in range(CONFIG["vqvae_epochs"]):
        total = 0.0
        vqvae.train()
        for batch in loader:
            batch = batch.to(CONFIG["device"])
            optimizer.zero_grad()
            _, loss = vqvae(batch)
            loss.backward()
            clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        epoch_losses.append(avg)
        print(f"Epoch {epoch+1}/{CONFIG['vqvae_epochs']} — Loss: {avg:.6f}")

    # Test MSE
    with torch.no_grad():
        recon, _ = vqvae(test_embs.to(CONFIG["device"]))
        mse = torch.mean((recon - test_embs.to(CONFIG["device"])).pow(2)).item()
    print(f"\n✅ Test reconstruction MSE: {mse:.6f}")
    print(f"🎯 Semantic fidelity: {(1-mse)*100:.2f}%\n")

    # Save loss curve
    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, CONFIG["vqvae_epochs"]+1), epoch_losses,
             marker="o", linewidth=2, markersize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("VQ-VAE Training Progress")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/loss_curve.png", dpi=300)
    plt.close()
    print("✅ Saved loss curve to artifacts/loss_curve.png\n")

    # Export artifacts
    print("💾 Exporting reconstruction test data…")
    # sample latents
    with torch.no_grad():
        sample_latents = vqvae.encoder(test_embs.to(CONFIG["device"]))
    torch.save(sample_latents.cpu(), "artifacts/sample_latents.pt")
    # originals
    torch.save(embs, "artifacts/original_embeddings.pt")
    with open("artifacts/original_sentences.txt", "w", encoding="utf-8") as f:
        for s in corpus:
            f.write(s.replace("\n", " ") + "\n")
    # config for downstream
    cfg_export = {
        "embedding_dimension": d,
        "vqvae_hidden_dim": CONFIG["vqvae_hidden_dim"],
        "vqvae_embedding_dim": CONFIG["vqvae_embedding_dim"],
        "vqvae_num_embeddings": CONFIG["vqvae_num_embeddings"]
    }
    with open("artifacts/vqvae_config.json", "w") as f:
        json.dump(cfg_export, f, indent=2)
    # FAISS & model
    if CONFIG["use_faiss_gpu"] and GPU_FAISS_AVAILABLE:
        cpu_back = index_gpu_to_cpu(index)
        faiss.write_index(cpu_back, "artifacts/pq_opq.index")
    else:
        faiss.write_index(index, "artifacts/pq_opq.index")
    torch.save(vqvae.state_dict(), "artifacts/vqvae.pt")
    # metrics CSV
    pd.DataFrame([{"test_mse": mse}]).to_csv("artifacts/evaluation_metrics.csv", index=False)
    print("✅ All artifacts exported to ./artifacts\n")

if __name__ == "__main__":
    main()
