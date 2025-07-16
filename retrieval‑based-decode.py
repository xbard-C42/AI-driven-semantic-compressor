# retrieval‑based-decode.py  — Retrieval‑based decoder instead of LLM
"""
Decode sample_latents.pt by:
  1. Passing each latent through the VQ‑VAE decoder to get a 384‑D embedding
  2. Searching your FAISS PQ+OPQ index for the closest original embedding
  3. Printing the corresponding original sentence
"""
import os
import json
import torch
import faiss
from semantic_compression_opq_vqvae_pipeline import VQVAE, CONFIG as PIPE_CFG

ART = "artifacts"
LATENT_PATH   = os.path.join(ART, "sample_latents.pt")
EMB_PATH      = os.path.join(ART, "original_embeddings.pt")
SENT_PATH     = os.path.join(ART, "original_sentences.txt")
INDEX_PATH    = os.path.join(ART, "pq_opq.index")
MODEL_WEIGHTS = os.path.join(ART, "vqvae.pt")
CFG_PATH      = os.path.join(ART, "vqvae_config.json")

# 1) Load latents & originals
latents = torch.load(LATENT_PATH)                # [N, 64]
orig_embs_np = torch.load(EMB_PATH).numpy().astype("float32")  # [10000, 384]
with open(SENT_PATH, "r", encoding="utf-8") as f:
    sentences = [l.strip() for l in f]

# 2) Load PQ+OPQ FAISS index
index = faiss.read_index(INDEX_PATH)

# 3) Load VQ‑VAE decoder
with open(CFG_PATH, "r") as f:
    cfg = json.load(f)
vqvae = VQVAE(
    input_dim=cfg["embedding_dimension"],
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
).to(PIPE_CFG["device"])
vqvae.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=PIPE_CFG["device"]))
vqvae.eval()

# 4) Decode & retrieve
print("\n🗄️  Retrieval-based reconstructions:\n")
with torch.no_grad():
    # Pass latents through decoder to get embeddings
    decoded_embs = vqvae.decoder(latents.to(PIPE_CFG["device"])).cpu().numpy().astype("float32")  # [N, 384]

# Search top-1 nearest neighbour
_, idxs = index.search(decoded_embs, k=1)  # idxs.shape = (N,1)

for i, (latent, [idx]) in enumerate(zip(latents, idxs)):
    print(f"[{i}] 🔎 Retrieved: “{sentences[idx]}”")
