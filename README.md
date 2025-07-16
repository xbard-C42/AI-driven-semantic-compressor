<div align="center">

# 🔬 Semantic Compression VQ‑VAE Pipeline

<div align="center">

![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=28&duration=3000&pause=1000&color=F07178&center=true&vCenter=true&width=900&lines=Compressing+Meaning+Not+Just+Data;Quantize+↔️+Reconstruct+via+LLMs;Revolutionizing+Semantic+Storage+%26+Retrieval)

**📍 Anywhere Data Meets Insight**  
**🎯 Mission: Semantic Compression + Generative Reconstruction**  
**💬 Motto: “Beyond Bits—Understanding.”**

</div>

---

## 🧠 The Grand Vision

> **What if meaning could be compressed like bytes?**  
>  
> Imagine storing vast corpora in a few hundred bits—and then bringing sentences back to life with an LLM whisper. That’s not sci‑fi; it’s what we’re building.

This pipeline unites discrete VQ‑VAE codebooks with FAISS OPQ+PQ indexing and state‑of‑the‑art LLM decoders—letting embeddings become storage, and storage become sentences again. 🤖✨

---

## ⚙️ Core Components

1. 📊 **pipeline.py**  
   **Input:** Raw text (Wikitext‑2) → SBERT embeddings  
   **Compression:** FAISS OPQ+PQ → VQ‑VAE discrete latents  
   **Output:** `artifacts/sample_latents.pt`, model checkpoints & metadata

2. 🤖 **decode.py**  
   **Load:** Quantized latents & config  
   **Reconstruct:** Cosine‑similarity fallback or LLMs (GPT‑4O, Claude‑2, Gemini‑1.5‑Flash)  
   **Output:** Human‑readable sentences

3. 📈 **evaluate.py**  
   **Metrics:** Cosine‑similarity, BERTScore, BLEU  
   **Report:** CSV, HTML & radar plot  
   **Visuals:** `artifacts/evaluation_radar_plot.png`

4. 📡 **llm_callers.py**  
   Unified wrappers for each model’s API—just set your `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GOOGLE_API_KEY`.

---

## 🚀 Quickstart

```bash
# 1. Clone & config
git clone https://github.com/xbard-C42/AI-driven-semantic-compressor-VQ-VAE-Pipeline.git
cd AI-driven-semantic-compressor-VQ-VAE-Pipeline

# 2. Create vqvae_config.json
cat > vqvae_config.json << 'EOF'
{
  "device": "cpu",
  "dataset_split": "train",
  "sample_size": 10000,
  "batch_size": 64,
  "test_fraction": 0.1,
  "pq_M": 8,
  "embedding_dimension": 384,
  "vqvae_hidden_dim": 128,
  "vqvae_embedding_dim": 64,
  "vqvae_num_embeddings": 512,
  "vqvae_learning_rate": 1e-3,
  "vqvae_epochs": 10
}
EOF

# 3. Install deps
pip install -r requirements.txt

# 4. Run the 3‑step pipeline
python pipeline.py     # ▶️ train & save latents
python decode.py       # ▶️ reconstruct with similarity/LLMs
python evaluate.py     # ▶️ metrics & radar plot





It now works as intended and my work is likely done. 18:28 16/07/2025. It's your's now!
