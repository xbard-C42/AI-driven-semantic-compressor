# decode.py
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import torch
from sentence_transformers import SentenceTransformer

from llm_callers import call_openai, call_claude, call_gemini

ARTIFACT_DIR = "artifacts"
LATENT_PATH  = f"{ARTIFACT_DIR}/sample_latents.pt"
EMB_PATH     = f"{ARTIFACT_DIR}/original_embeddings.pt"
SENT_PATH    = f"{ARTIFACT_DIR}/original_sentences.txt"
MODEL_PATH   = f"{ARTIFACT_DIR}/vqvae.pt"
CONFIG_PY    = "semantic_compression_opq_vqvae_pipeline"  # module name for VQVAE

if not os.path.exists(LATENT_PATH):
    raise FileNotFoundError("Run pipeline first to generate sample_latents.pt")

# Load latents
compressed = torch.load(LATENT_PATH)

# Load model & config
import semantic_compression_opq_vqvae_pipeline as pipeline_mod
cfg = pipeline_mod.CONFIG
from semantic_compression_opq_vqvae_pipeline import VQVAE
vqvae = VQVAE(
    input_dim=compressed.shape[1],
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
)
vqvae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
vqvae.eval()

# Load originals for fallback
try:
    originals = torch.load(EMB_PATH)
    with open(SENT_PATH, "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f]
except:
    originals = sentences = None

sbert = SentenceTransformer("all-MiniLM-L6-v2")

def similarity_match(vec):
    sims = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), originals)
    idx = int(torch.argmax(sims))
    return sentences[idx]

for i, lat in enumerate(compressed[:10]):
    with torch.no_grad():
        decoded_emb = vqvae.decoder(lat.unsqueeze(0)).squeeze(0)

    # Primary: OpenAI
    try:
        sent = call_openai(decoded_emb.tolist())
    except Exception:
        # Fallback: Claude
        try:
            sent = call_claude(decoded_emb.tolist())
        except Exception:
            # Fallback: Gemini
            try:
                sent = call_gemini(decoded_emb.tolist())
            except Exception:
                # Final: similarity
                sent = similarity_match(decoded_emb) if originals is not None else "[unreconstructable]"
    print(f"[{i}] 🔄 Reconstructed: “{sent}”")
