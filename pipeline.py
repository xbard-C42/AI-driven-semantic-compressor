# pipeline.py
import os
import json
import random

import torch
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 1. Config Loading
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# 2. Data Preparation
class TextDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# 3. VQ-VAE Model Definition
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, D)
        flat_inputs = inputs.view(-1, self.embedding_dim)  # (B, D)
        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight**2, dim=1)
        )  # (B, num_embeddings)
        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        quantized = self.embeddings(encoding_indices).view_as(inputs)
        e_latent_loss = torch.mean((quantized.detach() - inputs).pow(2))
        q_latent_loss = torch.mean((quantized - inputs.detach()).pow(2))
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # straight‑through
        quantized = inputs + (quantized - inputs).detach()
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

# 4. Model Builder
def build_model(config):
    return VQVAE(
        input_dim=config["embedding_dimension"],
        hidden_dim=config["vqvae_hidden_dim"],
        embedding_dim=config["vqvae_embedding_dim"],
        num_embeddings=config["vqvae_num_embeddings"]
    )

# 5. Pipeline
def main():
    cfg = load_config("vqvae_config.json")
    os.makedirs("artifacts", exist_ok=True)

    print("🎪 Starting Enhanced Semantic Compression Pipeline\n")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=cfg["dataset_split"])
    sentences = [s for s in ds["text"] if len(s.split()) > 5]
    if len(sentences) < cfg["sample_size"]:
        raise ValueError(f"Not enough sentences ({len(sentences)} < {cfg['sample_size']})")
    random.seed(42)
    corpus = random.sample(sentences, k=cfg["sample_size"])
    print(f"✅ Sampled {len(corpus)} sentences.\n")

    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=cfg["device"])
    embeddings = sbert.encode(corpus, batch_size=cfg["batch_size"], show_progress_bar=True)
    embeddings = torch.from_numpy(np.array(embeddings)).float()
    print(f"✅ Generated embeddings: {embeddings.shape}\n")

    # Save originals for later
    torch.save(embeddings, "artifacts/original_embeddings.pt")
    with open("artifacts/original_sentences.txt", "w", encoding="utf-8") as f:
        for sent in corpus:
            f.write(sent.replace("\n", " ") + "\n")
    print("✅ Saved original embeddings & sentences.\n")

    train_embs, test_embs = train_test_split(
        embeddings, test_size=cfg["test_fraction"], random_state=42
    )
    print(f"✅ Train/Test split: {len(train_embs)}/{len(test_embs)}\n")

    # FAISS OPQ + PQ
    print("🔍 Training FAISS OPQ + PQ index...")
    all_np = embeddings.numpy().astype("float32")
    d = all_np.shape[1]
    cpu_index = faiss.IndexPreTransform(
        faiss.OPQMatrix(d, cfg["pq_M"]),
        faiss.IndexFlatL2(d)
    )
    cpu_index.train(all_np)
    cpu_index.add(all_np)
    print("✅ FAISS index ready!\n")

    # Build & train VQ‑VAE
    vqvae = build_model(cfg).to(cfg["device"])
    optimizer = optim.Adam(vqvae.parameters(), lr=cfg["vqvae_learning_rate"])
    loader = DataLoader(TextDataset(train_embs), batch_size=cfg["batch_size"], shuffle=True)

    for epoch in range(cfg["vqvae_epochs"]):
        vqvae.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(cfg["device"])
            optimizer.zero_grad()
            _, loss = vqvae(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg['vqvae_epochs']} — Loss: {avg:.4f}")
    print("\n✅ Training complete!")

    torch.save(vqvae.state_dict(), "artifacts/vqvae.pt")
    print("✅ Model saved to artifacts/vqvae.pt\n")

    # Export sample latents for decoding
    vqvae.eval()
    with torch.no_grad():
        test_tensor = test_embs.to(cfg["device"])
        z_e = vqvae.encoder(test_tensor)
        z_q, _ = vqvae.quantizer(z_e)
    sample_latents = z_q.cpu()[:10]
    torch.save(sample_latents, "artifacts/sample_latents.pt")
    print("✅ Saved sample latents to artifacts/sample_latents.pt\n")

if __name__ == "__main__":
    main()
