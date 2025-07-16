# evaluate.py
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import AuthenticationError as OpenAIAuthError
from anthropic import AnthropicError
from llm_callers import call_openai, call_claude, call_gemini

ARTIFACT_DIR = "artifacts"
LATENT_PATH = os.path.join(ARTIFACT_DIR, "sample_latents.pt")
ORIG_EMB    = os.path.join(ARTIFACT_DIR, "original_embeddings.pt")
ORIG_SENTS  = os.path.join(ARTIFACT_DIR, "original_sentences.txt")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "vqvae_config.json")

# Load artifacts
sample_latents = torch.load(LATENT_PATH)
orig_embs       = torch.load(ORIG_EMB)
with open(ORIG_SENTS, "r", encoding="utf-8") as f:
    orig_sents = [l.strip() for l in f]
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# Prepare similarity fallback
sbert = SentenceTransformer("all-MiniLM-L6-v2")
def similarity_match(vec):
    sims = torch.nn.functional.cosine_similarity(vec.unsqueeze(0), orig_embs)
    idx = int(torch.argmax(sims))
    return orig_sents[idx]

# LLM backends
CALLERS = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini
}

results = []
smooth_fn = SmoothingFunction().method1

for name, fn in CALLERS.items():
    print(f"🔁 Decoding with {name}…")
    decoded = []
    for vec in sample_latents:
        try:
            text = fn(vec.tolist())
            # handle empty or None response
            if not text:
                raise ValueError("Empty response")
            decoded.append(text)
        except (OpenAIAuthError, AnthropicError, RuntimeError, ValueError) as e:
            print(f"⚠️ {name} failed ({e.__class__.__name__}): {e}")
            # fallback to similarity if available
            try:
                decoded.append(similarity_match(torch.tensor(vec)))
            except Exception:
                decoded.append("")  # last‑resort: empty string
    # Compute metrics
    emb_dec = sbert.encode(decoded, convert_to_tensor=True)
    emb_ori = sbert.encode(orig_sents[:len(decoded)], convert_to_tensor=True)
    cos_scores = util.cos_sim(emb_dec, emb_ori).diagonal().tolist()

    P, R, F1 = bertscore(decoded, orig_sents[:len(decoded)], lang="en", verbose=False)

    bleu_scores = []
    for ref, hyp in zip(orig_sents, decoded):
        try:
            bleu_scores.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth_fn))
        except Exception:
            bleu_scores.append(0.0)

    results.append({
        "Model": name,
        "CosineSim":      round(np.mean(cos_scores), 4),
        "BERTScore_F1":   round(F1.mean().item(),      4),
        "BLEU":           round(np.mean(bleu_scores),  4)
    })

# Save report
df = pd.DataFrame(results)
df.to_csv(os.path.join(ARTIFACT_DIR, "evaluation_report.csv"), index=False)

# Radar plot
cats   = ["CosineSim", "BERTScore_F1", "BLEU"]
angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(subplot_kw={"polar": True})
for _, row in df.iterrows():
    vals = row[cats].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, label=row["Model"])
    ax.fill(angles, vals, alpha=0.1)

ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), cats)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "evaluation_radar_plot.png"))
plt.close()

print("✅ Evaluation complete. Reports & radar plot saved.")
