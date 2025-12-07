The Model Architecture phase involves:
 - Implementing self-attention mechanisms with trainable Q, K, V projections
 - Building transformer blocks with attention + feed-forward + normalization
 - Stacking multiple blocks (12 for GPT-2 small)
 - Adding embedding layers (token + position)
 - Connecting all components into a cohesive GPT model
---

## Model Configurations

The book implements **GPT-2 small** but discusses various sizes:

| Model | Parameters | Layers | Heads | Embedding Dim |
|-------|-----------|--------|-------|---------------|
| GPT-2 Small | 124M | 12 | 12 | 768 |
| GPT-2 Medium | 355M | 24 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 20 | 1280 |
| GPT-2 XL | 1.5B | 48 | 25 | 1600 |

---

The most common ML architectures and their use cases, including:
 - 
 - 
 - 