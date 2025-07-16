# decode.py
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from semantic_compression_opq_vqvae_pipeline import VQVAE
from llm_callers import call_openai, call_claude, call_gemini

ARTIFACT_DIR = "artifacts"
LATENT_PATH = os.path.join(ARTIFACT_DIR, "sample_latents.pt")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "vqvae_config.json")

if not os.path.exists(LATENT_PATH):
    raise FileNotFoundError("Run pipeline first to generate sample_latents.pt")

# Load config
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# Load latents
compressed = torch.load(LATENT_PATH)

# Instantiate VQ-VAE decoder
vqvae = VQVAE(
    input_dim=cfg["embedding_dimension"],
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
)
vqvae.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "vqvae.pt"),
                                 map_location="cpu"))
vqvae.eval()

# Load originals for fallback
try:
    originals = torch.load(os.path.join(ARTIFACT_DIR, "original_embeddings.pt"))
    with open(os.path.join(ARTIFACT_DIR, "original_sentences.txt"), "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f]
except:
    originals, sentences = None, None

sbert = SentenceTransformer("all-MiniLM-L6-v2")

def similarity_match(vec):
    sims = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), originals)
    idx = int(torch.argmax(sims))
    return sentences[idx]

BACKEND = os.getenv("SEMANTIC_RECON_BACKEND", "openai").lower()
CALLER = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini
}.get(BACKEND, call_openai)

for i, vec in enumerate(compressed[:10]):
    with torch.no_grad():
        decoded = vqvae.decoder(vec.unsqueeze(0)).squeeze(0)
    if originals is not None:
        sent = similarity_match(decoded)
        print(f"[{i}] 🔍 Similarity: “{sent}”")
    else:
        sentence = CALLER(decoded.tolist())
        print(f"[{i}] 🤖 LLM: “{sentence}”")
