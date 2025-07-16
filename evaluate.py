# evaluate.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from llm_callers import call_openai, call_claude, call_gemini

os.makedirs("artifacts", exist_ok=True)

CALLERS = {
    "openai": call_openai,
    "claude": call_claude,
    "gemini": call_gemini
}

# Load data
sample_latents = torch.load("artifacts/sample_latents.pt")
original_embeddings = torch.load("artifacts/original_embeddings.pt")
with open("artifacts/original_sentences.txt", "r", encoding="utf-8") as f:
    original_sentences = [l.strip() for l in f]

sbert = SentenceTransformer("all-MiniLM-L6-v2")
results = []
df_list = []

for model_name, decoder in CALLERS.items():
    print(f"🔁 Decoding with {model_name}...")
    decoded = [decoder(vec.tolist()) for vec in sample_latents]

    # Metrics
    emb_dec = sbert.encode(decoded, convert_to_tensor=True)
    emb_ori = sbert.encode(original_sentences[: len(decoded) ], convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_dec, emb_ori).diagonal().tolist()

    P, R, F1 = bertscore(decoded, original_sentences[: len(decoded)], lang="en", verbose=False)

    bleu_scores = []
    smoother = SmoothingFunction().method1
    for ref, hyp in zip(original_sentences, decoded):
        try:
            bleu_scores.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoother))
        except:
            bleu_scores.append(0.0)

    results.append({
        "Model": model_name,
        "CosineSimilarity": round(np.mean(cosine_scores), 4),
        "BERTScore_F1": round(F1.mean().item(), 4),
        "BLEU": round(np.mean(bleu_scores), 4)
    })

# Export
df = pd.DataFrame(results)
df.to_csv("artifacts/evaluation_report.csv", index=False)
with open("artifacts/evaluation_report.html", "w", encoding="utf-8") as f:
    f.write("<h2>Evaluation Report</h2>" + df.to_html(index=False))
print("✅ Reports saved.")

# Radar plot
def radar_plot(dataframe):
    categories = list(dataframe.columns[1:])
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    for i, row in dataframe.iterrows():
        vals = row.drop("Model").tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=row["Model"])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("artifacts/evaluation_radar_plot.png")
    print("✅ Radar plot saved.")

radar_plot(df)
