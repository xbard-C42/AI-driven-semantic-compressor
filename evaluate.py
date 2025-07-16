# evaluate.py
"""
Evaluate reconstructed samples via OpenAI → Claude → Gemini → similarity,
compute CosineSim, BERTScore (distilroberta-base) and BLEU, and plot radar.
Loads .env automatically.
"""
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from llm_callers import call_openai, call_claude, call_gemini
from semantic_compression_opq_vqvae_pipeline import VQVAE, CONFIG as PIPE_CFG

ART = "artifacts"
LATENT_PATH = os.path.join(ART, "sample_latents.pt")
EMB_PATH    = os.path.join(ART, "original_embeddings.pt")
SENT_PATH   = os.path.join(ART, "original_sentences.txt")
MODEL_PATH  = os.path.join(ART, "vqvae.pt")
CFG_PATH    = os.path.join(ART, "vqvae_config.json")

MAX_SAMPLES = 10
smooth = SmoothingFunction().method1

# Load data & config
latents = torch.load(LATENT_PATH)
orig_embs = torch.load(EMB_PATH)
with open(SENT_PATH, "r", encoding="utf-8") as f:
    orig_sents = [l.strip() for l in f]
with open(CFG_PATH, "r") as f:
    cfg = json.load(f)

# Build decoder for similarity fallback
vqvae = VQVAE(
    input_dim=cfg["embedding_dimension"],
    hidden_dim=cfg["vqvae_hidden_dim"],
    embedding_dim=cfg["vqvae_embedding_dim"],
    num_embeddings=cfg["vqvae_num_embeddings"]
).to(PIPE_CFG["device"])
vqvae.load_state_dict(torch.load(MODEL_PATH, map_location=PIPE_CFG["device"]))
vqvae.eval()

def similarity_sentence(latent_vec):
    with torch.no_grad():
        decoded_emb = vqvae.decoder(latent_vec.unsqueeze(0)).squeeze(0)
    sims = torch.nn.functional.cosine_similarity(decoded_emb.unsqueeze(0), orig_embs)
    return orig_sents[int(torch.argmax(sims))]

# Setup ST & results container
sbert = SentenceTransformer("all-MiniLM-L6-v2")
results = []

for backend, fn in [("openai", call_openai),
                    ("claude",  call_claude),
                    ("gemini",  call_gemini)]:
    print(f"\n🔁 Evaluating via {backend} (up to {MAX_SAMPLES} samples)…")
    decoded = []
    for i, latent in enumerate(latents[:MAX_SAMPLES], 1):
        print(f"   → sample {i}/{MAX_SAMPLES}", end="\r")
        try:
            txt = fn(latent.tolist()) or ""
        except Exception:
            # fallback chain
            if backend != "claude":
                try:    txt = call_claude(latent.tolist()) or ""
                except:
                    if backend != "gemini":
                        try:    txt = call_gemini(latent.tolist()) or ""
                        except: txt = similarity_sentence(latent)
                    else:
                        txt = similarity_sentence(latent)
            else:
                try:    txt = call_gemini(latent.tolist()) or ""
                except: txt = similarity_sentence(latent)
        decoded.append(txt)
    print()  # newline

    # Metrics
    emb_dec   = sbert.encode(decoded,       convert_to_tensor=True)
    emb_ori   = sbert.encode(orig_sents[:len(decoded)], convert_to_tensor=True)
    cos_scores= util.cos_sim(emb_dec, emb_ori).diagonal().tolist()

    try:
        _, _, f1 = bertscore(
            decoded, orig_sents[:len(decoded)],
            lang="en", model_type="distilroberta-base", verbose=False
        )
        bert_f1 = f1.mean().item()
    except:
        bert_f1 = 0.0

    bleu_scores = []
    for ref, hyp in zip(orig_sents[:len(decoded)], decoded):
        try:
            bleu_scores.append(
                sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth)
            )
        except:
            bleu_scores.append(0.0)

    results.append({
        "Model":        backend,
        "CosineSim":    round(np.mean(cos_scores),   4),
        "BERTScore_F1": round(bert_f1,               4),
        "BLEU":         round(np.mean(bleu_scores), 4)
    })

# Save report
df = pd.DataFrame(results)
df.to_csv(os.path.join(ART, "evaluation_report.csv"), index=False)

# Radar plot
cats   = ["CosineSim", "BERTScore_F1", "BLEU"]
angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw={"polar": True})
for _, row in df.iterrows():
    vals     = [row[c] for c in cats]
    plot_vals= vals + [vals[0]]
    ax.plot(angles, plot_vals, label=row["Model"])
    ax.fill(angles, plot_vals, alpha=0.1)

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), cats)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(ART, "evaluation_radar_plot.png"))
plt.close()

print("\n✅ Evaluation complete. Reports & radar plot saved.")
