# Semantic Compression VQ-VAE Pipeline

Train a VQ-VAE on sentence embeddings, decode quantized latents via LLMs, and compare reconstruction quality.

## Setup

1. Clone this repo.
2. Create a `vqvae_config.json` in the root. Example:

   ```json
   {
     "device": "cpu",
     "dataset_split": "train",
     "sample_size": 10000,
     "batch_size": 64,
     "test_fraction": 0.1,
     "pq_M": 8,
     "vqvae_hidden_dim": 128,
     "vqvae_embedding_dim": 64,
     "vqvae_num_embeddings": 512,
     "vqvae_learning_rate": 1e-3,
     "vqvae_epochs": 10
   }
