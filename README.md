# NanoGPT

A minimal and efficient implementation of a Transformer-based language model, inspired by GPT. This project is built from scratch using PyTorch, with support for FlashAttention, Cosine LR Decay, Checkpointing, and Distributed Data Parallel (DDP) training.

The resultant model is not the best but a great leap in terms of machine learning.

THIS IS A TEST APP
https://chatbot-transformer.streamlit.app/



## 🚀 Model Overview

This GPT model follows the architecture of the original Transformer, as introduced in [Attention is All You Need](https://doi.org/10.48550/arXiv.1706.03762).
The basic System design is inspired by [ChatGPT2](https://github.com/openai/gpt-2) which was implemented in Tensorflow.
Introduction of new concept like Flash Attention  have decreased Training time as compared to initial GPT2 Model.
Training date used is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer) which is much better that OpenWebText used by Original ChatGPT2.


![image](https://github.com/user-attachments/assets/726cddb4-f620-4385-a147-add57e7a4ba1)




### 🔧 Transformer Block

Each block in the model is structured as:

```
x → LayerNorm → MultiHeadSelfAttention → Add & Norm → FeedForward → Dropout → Add & Norm
```

Mathematically:

$$
\text{TransformerBlock}(x) = x + \text{Dropout}\left( \text{FFN}\left( \text{LayerNorm}\left( x + \text{MultiHeadSelfAttention}(\text{LayerNorm}(x)) \right) \right) \right)
$$

Where:
- **LayerNorm** normalizes the input.
- **MultiHeadSelfAttention** captures contextual relationships.
- **FFN** is a position-wise feed-forward network.
- **Dropout** improves generalization.

### Architecture Flow

The model processes input tokens as follows:

```
x_0 = TokenEmbedding(input)
x_{i+1} = TransformerBlock(x_i),  for i = 0, ..., N-1
\hat{y} = Softmax(Linear(x_N))
```

###  Objective

The model is trained using the **causal language modeling** loss:

```
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
```

##  Training Features

- **Cosine Learning Rate Decay**:

  ```
  \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}} \pi\right)\right)
  ```

- **FlashAttention** for faster and memory-efficient training.
- **Checkpointing** to save and resume training.
- **Distributed Data Parallel (DDP)** support for multi-GPU scalability.

