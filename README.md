# NanoGPT

A minimal and efficient implementation of a Transformer-based language model, inspired by the GPT architecture. This project is built from scratch using **PyTorch**, designed to be a clean, educational, yet capable codebase.

It incorporates modern training optimizations including **FlashAttention**, **Cosine Learning Rate Decay**, **Checkpointing**, and **Distributed Data Parallel (DDP)** training.

> **Note:** While this is a test implementation and not a State-of-the-Art (SOTA) foundation model, it represents a significant structural leap in understanding efficient LLM training.

### Live Demo
Try the test application here: **[Chatbot Transformer App](https://chatbot-transformer.streamlit.app/)**

---

## Model Overview

This model adheres to the original Transformer architecture introduced in *[Attention is All You Need](https://doi.org/10.48550/arXiv.1706.03762)*. The system design draws inspiration from OpenAI's [GPT-2](https://github.com/openai/gpt-2), originally implemented in TensorFlow, but modernized here for PyTorch.

### Key Improvements
| Feature | Description |
| :--- | :--- |
| **FlashAttention** | Drastically reduces training time and memory usage compared to the standard attention mechanism used in early GPT-2 implementations. |
| **FineWeb-Edu** | Trained on the [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer), a high-quality educational dataset superior to the OpenWebText corpus used by the original GPT-2. |

![Model Architecture Diagram](https://github.com/user-attachments/assets/726cddb4-f620-4385-a147-add57e7a4ba1)

---

## 🔧 Architecture Details

### The Transformer Block
The core component of the model is the Transformer Block. Each block processes the input stream through the following layers:

**Flow:**
`x → LayerNorm → MultiHeadSelfAttention → Add & Norm → FeedForward → Dropout → Add & Norm`

**Mathematical Formulation:**
$$
\text{Block}(x) = x + \text{Dropout}\left( \text{FFN}\left( \text{LayerNorm}\left( x + \text{MHSA}(\text{LayerNorm}(x)) \right) \right) \right)
$$

Where:
* **LayerNorm:** Normalizes input across features to stabilize training.
* **MHSA (Multi-Head Self-Attention):** Captures long-range dependencies and contextual relationships.
* **FFN (Feed-Forward Network):** A position-wise network processing each token independently.
* **Dropout:** Randomly zeroes elements to prevent overfitting.

### Processing Pipeline
The model processes a sequence of input tokens as follows:

$$
\begin{aligned}
x_0 &= \text{TokenEmbedding}(\text{input}) + \text{PositionalEmbedding}(\text{indices}) \\
x_{i+1} &= \text{TransformerBlock}(x_i), \quad \text{for } i = 0, \dots, N-1 \\
\hat{y} &= \text{Softmax}(\text{Linear}(x_N))
\end{aligned}
$$

### Training Objective
The model is optimized using **Causal Language Modeling (CLM)** loss, maximizing the probability of the next token given previous tokens:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

---

## Training Features

The training loop implements several advanced features to ensure stability and speed:

* **Cosine Learning Rate Decay:**
    Smoothly reduces the learning rate to improve convergence.
    $$
    \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}} \pi\right)\right)
    $$

* **FlashAttention:** Utilized for fast and memory-efficient exact attention.
* **Checkpointing:** Automatic saving of model states to resume training seamlessly.
* **Distributed Data Parallel (DDP):** Supports scaling training across multiple GPUs.
