# Architecture Overview

**Test-Model-Thing (TMT)** is an experimental, tokenizer-free, byte-level recurrent language model implemented in pure NumPy. It combines continuous streaming, latent-space prediction (JEPA style), and Recurrent Trace Units (RTUs) for constant-memory sequence modeling.

---

## Key Capabilities

1. **Tokenizer-Free Byte I/O ($V = 256$)**:
   - Processes raw bytes ($0$ to $255$) directly without BPE tokenizers, sentencepiece, or subword dictionaries.
   - Eliminates massive vocabulary embedding sinks (e.g., $32\text{k}\times 4096$ matrices in LLMs).
   - Naturally multimodal: can ingest and emit arbitrary binary formats, text, or byte streams.

2. **$O(1)$ State Space via Recurrent Trace Units (RTUs)**:
   - Does not require a quadratic context window or an expanding KV cache.
   - Each layer maintains internal memory state vectors:
     $$\mathbf{s}_t = \sigma(\mathbf{w}_{\text{decay}}) \odot \mathbf{s}_{t-1} + \mathbf{x}_t$$
   - Old memories decay smoothly over time via learned, channel-wise decay factors.

3. **JEPA-Style Latent Space Prediction**:
   - The model does not solely optimize next-token cross-entropy; it simultaneously predicts the next latent representation:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{var}} + \mathcal{L}_{\text{pred\_mse}} + \mathcal{L}_{\text{ce}} + \mathcal{L}_{\text{stop\_mse}}$$
   - **Variance Regularization**: Prevents latent representation collapse across sequence steps.
   - **Stop Token Prediction**: Predicts sequence termination without dedicated delimiter tokens.

4. **Continual Test-Time Adaptation**:
   - The model can update its weights continuously during interactive chatting, training on user prompts and its own outputs in real-time.
   - Readonly and frozen modes allow fine-grained control over whether state or weights advance.

---

## Model Architecture

```
Byte Input (0..255)
       │
       ▼
   Embedding (256 -> dim)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
 ┌─────────────┐                          ┌─────────────┐
 │   Layer 0   │                          │   Layer L   │
 │             │                          │             │
 │  s = decay*s│                          │  s = decay*s│
 │      + enc  │                          │      + enc  │
 │  norm(s)    │───► [Linear + SiLU] ───► │             │
 └─────────────┘                          └─────────────┘
       │                                         │
       ▼                                         ▼
   Residual X ───────────────────────────► Final Latent X
                                                 │
                                ┌────────────────┴────────────────┐
                                ▼                                 ▼
                         Decoder (dim->256)                Stop (dim->1)
                                │                                 │
                                ▼                                 ▼
                          Next Byte Logits                Stop Probability
```

---

## Parameter Calculation

For a model with latent dimension $D$ and $L$ layers:
- **Encoder**: $256 \times D$
- **Per Layer**: $D^2 + 3D$ (weights $D\times D$, layer norm scale $D$, layer norm bias $D$, decay logits $D$)
- **Decoder**: $256 \times D + 256$ (weights + bias)
- **Stop Head**: $D + 1$ (weight + bias)

$$\text{Total Params} = 256 D + L(D^2 + 3D) + 256 D + 256 + D + 1$$

For default hyperparameters ($D = 512, L = 16$):
$$\text{Total Params} = 4,481,793 \approx 4.48\text{M}$$
