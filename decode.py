# decode.py
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline import VQVAE, load_config

# Prepare
os.makedirs("artifacts", exist_ok=True)

# Load compressed latents
latent_path = "artifacts/sample_latents.pt"
if not os.path.exists(latent_path):
    raise FileNotFoundError(f"Missing file: {latent_path}. Run pipeline.py first.")
compressed_vectors = torch.load(latent_path)

# Load originals (for similarity fallback)
try:
    original_embeddings = torch.load("artifacts/original_embeddings.pt")
    with open("artifacts/original_sentences.txt", "r", encoding="utf-8") as f:
        original_sentences = [l.strip() for l in f]
except Exception as e:
    print(f"⚠️ Warning: Failed to load originals: {e}")
    original_embeddings = None
    original_sentences = None

# Load VQ‑VAE
cfg = load_config("vqvae_config.json")
dim = compressed_vectors.shape[1]
vqvae = VQVAE(
    input_dim=dim,
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
)
vqvae.load_state_dict(torch.load("artifacts/vqvae.pt", map_location="cpu"))
vqvae.eval()

# Decode back to embedding space
with torch.no_grad():
    decoded_vectors = vqvae.decoder(compressed_vectors)
    torch.save(decoded_vectors, "artifacts/decoded_vectors.pt")

# Similarity Fallback
def similarity_match(decoded_vec, originals, sentences):
    sims = torch.nn.functional.cosine_similarity(decoded_vec.unsqueeze(0), originals)
    idx = int(torch.argmax(sims))
    return sentences[idx], float(sims[idx])

# LLM Backends
from llm_callers import call_openai, call_claude, call_gemini
BACKEND = os.getenv("SEMANTIC_RECON_BACKEND", "openai").lower()
CALLER = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini
}.get(BACKEND, call_openai)

# Sentence encoder for similarity
sbert = SentenceTransformer("all-MiniLM-L6-v2")

for i, vec in enumerate(decoded_vectors[:10]):
    try:
        if original_embeddings is not None:
            sent, sim = similarity_match(vec, original_embeddings, original_sentences)
            print(f"[{i}] 🔍 Similarity match: “{sent}” (cos={sim:.3f})")
        else:
            sentence = CALLER(vec.tolist())
            print(f"[{i}] 🤖 LLM Reconstruct: “{sentence}”")
    except Exception as e:
        print(f"[{i}] ⚠️ Error: {e}")
