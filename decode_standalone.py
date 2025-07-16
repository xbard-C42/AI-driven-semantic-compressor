# decode_standalone.py
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import json
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from llm_callers import call_openai, call_claude, call_gemini

# Standalone VQVAE classes (no imports from pipeline)
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
        dists = (flat_z.pow(2).sum(1, keepdim=True)
                 - 2 * flat_z @ self.embedding.weight.t()
                 + self.embedding.weight.pow(2).sum(1))
        indices = torch.argmin(dists, dim=1)
        quantized = self.embedding(indices).view_as(z)
        e_loss = torch.mean((quantized.detach() - z).pow(2))
        q_loss = torch.mean((quantized - z.detach()).pow(2))
        loss = q_loss + self.commitment_cost * e_loss
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

# Load config and data
with open("artifacts/vqvae_config.json", "r") as f:
    cfg = json.load(f)

print(f"🔍 Config: {cfg}")

# Create model with correct dimensions
vqvae = VQVAE(
    input_dim=cfg["embedding_dimension"],
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
)

print(f"🔍 VQVAE created with input_dim={cfg['embedding_dimension']}")

# Load trained weights
vqvae.load_state_dict(torch.load("artifacts/vqvae.pt", map_location="cpu"))
vqvae.eval()

print("✅ Model loaded successfully!")

# Load test data
compressed = torch.load("artifacts/sample_latents.pt")
originals = torch.load("artifacts/original_embeddings.pt")
with open("artifacts/original_sentences.txt", "r", encoding="utf-8") as f:
    sentences = [l.strip() for l in f]

def similarity_match(vec):
    sims = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), originals)
    idx = int(torch.argmax(sims))
    return sentences[idx]

# Test reconstruction
for i, lat in enumerate(compressed[:10]):
    with torch.no_grad():
        decoded_emb = vqvae.decoder(lat.unsqueeze(0)).squeeze(0)

    # Try LLMs with fallbacks
    try:
        sent = call_openai(decoded_emb.tolist())
    except Exception:
        try:
            sent = call_claude(decoded_emb.tolist())
        except Exception:
            try:
                sent = call_gemini(decoded_emb.tolist())
            except Exception:
                sent = similarity_match(decoded_emb)
    
    print(f"[{i}] 🔄 Reconstructed: \"{sent}\"")