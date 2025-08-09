# slm
A small language model implementation

Consider a character-level language model built on a multi-layer state space model backbone, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through multiple sequential SSM layers, with final softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t]
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. Each SSM layer processes its inputs through the standard state space model formulation detailed in the SSM repository.

For the final layer, the SSM outputs logits over the vocabulary, followed by softmax normalization to produce probability distributions over the character vocabulary:

$$
\begin{align*}
P &= \frac{\exp(Z)}{\sum_c \exp(Z_c)}
\end{align*}
$$

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized:

$$
\begin{align*}
\mathcal{L} &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}}
\end{align*}
$$

Gradients flow backward through the architecture using standard backpropagation through time (BPTT) with the chain rule. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling combined with nucleus sampling:

$$
P_{\tau}(c) = \frac{\exp(Z_c / \tau)}{\sum_{c'} \exp(Z_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

Nucleus sampling further improves generation quality by restricting sampling to the nucleus of the probability distribution. For a given threshold $p \in (0,1)$, only tokens with cumulative probability $\leq p$ (when sorted in descending order by probability) are considered for sampling. This eliminates low-probability tokens that could lead to incoherent text, while preserving diversity within the most likely candidates.

The AdamW optimizer with weight decay is used for training, maintaining exponential moving averages for all parameters.

The implementation leverages CUDA for parallel computation across batch and sequence dimensions, with efficient kernel implementations for embedding lookup, Swish activation, softmax normalization, and gradient accumulation. Character sequences are extracted from text corpora through random position sampling, providing diverse training contexts.

## How to run
```
sudo apt update
sudo apt install clang time libcurl4-openssl-dev nvidia-cuda-toolkit
git submodule init
git submodule update
make run -j 4
```