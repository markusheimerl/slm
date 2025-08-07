# slm
A small language model implementation

Consider a character-level language model built on a multi-layer state space model backbone, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through multiple sequential SSM layers, with final softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t]
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. Each SSM layer processes its inputs through the standard state space model formulation:

$$
\begin{align*}
H_t^{(i)} &= X_t^{(i)}B_i^T + H_{t-1}^{(i)}A_i^T \\
O_t^{(i)} &= H_t^{(i)}\sigma(H_t^{(i)}) \\
Y_t^{(i)} &= O_t^{(i)}C_i^T + X_t^{(i)}D_i^T
\end{align*}
$$

The state transition matrix $A_i$ captures temporal dependencies, input matrix $B_i$ maps current inputs to state updates, output matrix $C_i$ projects nonlinearly activated states to outputs, and feedthrough matrix $D_i$ provides direct input-output connections. Each SSM layer processes the outputs from the previous layer:

For intermediate layers (i < final):
$$
\begin{align*}
X^{(i+1)} &= Y^{(i)}
\end{align*}
$$

For the final layer, the SSM outputs logits over the vocabulary:

$$
\begin{align*}
L &= Y^{(f)} \\
P &= \frac{\exp(L)}{\sum_c \exp(L_c)}
\end{align*}
$$

The swish activation $z\sigma(z)$ is applied within the SSM layers for nonlinearity, followed by softmax normalization to produce probability distributions over the character vocabulary.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
L &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}}
\end{align*}
$$

The gradient flows backward through each layer following the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial L_t} &= P_t - \mathbf{1}_{y_t}
\end{align*}
$$

The gradient then flows backward through each SSM layer following standard BPTT with Swish derivatives:

$$
\begin{align*}
\frac{\partial L}{\partial C_i} &= \sum_t (\frac{\partial L}{\partial Y_t^{(i)}})^T O_t^{(i)} \\
\frac{\partial L}{\partial D_i} &= \sum_t (\frac{\partial L}{\partial Y_t^{(i)}})^T X_t^{(i)} \\
\frac{\partial L}{\partial O_t^{(i)}} &= (\frac{\partial L}{\partial Y_t^{(i)}})C_i \\
\frac{\partial L}{\partial H_t^{(i)}} &= \frac{\partial L}{\partial O_t^{(i)}} \odot [\sigma(H_t^{(i)}) + H_t^{(i)}\sigma(H_t^{(i)})(1-\sigma(H_t^{(i)}))] + (\frac{\partial L}{\partial H_{t+1}^{(i)}})A_i \\
\frac{\partial L}{\partial A_i} &= \sum_t (\frac{\partial L}{\partial H_t^{(i)}})^T H_{t-1}^{(i)} \\
\frac{\partial L}{\partial B_i} &= \sum_t (\frac{\partial L}{\partial H_t^{(i)}})^T X_t^{(i)} \\
\frac{\partial L}{\partial W_E[c]} &= \sum_{\substack{t,b \\ X_{t,b}=c}} \left(B_1^T\frac{\partial L}{\partial H_t^{(1)}} + D_1^T\frac{\partial L}{\partial Y_t^{(1)}}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling combined with nucleus sampling:

$$
P_{\tau}(c) = \frac{\exp(L_c / \tau)}{\sum_{c'} \exp(L_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

Nucleus sampling further improves generation quality by restricting sampling to the nucleus of the probability distribution. For a given threshold $p \in (0,1)$, only tokens with cumulative probability $\leq p$ (when sorted in descending order by probability) are considered for sampling. This eliminates low-probability tokens that could lead to incoherent text, while preserving diversity within the most likely candidates.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A_i, B_i, C_i, D_i, W_E\}$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages CUDA for parallel computation across batch and sequence dimensions, with efficient kernel implementations for embedding lookup, Swish activation, softmax normalization, and gradient accumulation. Character sequences are extracted from text corpora through random position sampling, providing diverse training contexts.

## How to run
```
sudo apt update
sudo apt install clang time libcurl4-openssl-dev nvidia-cuda-toolkit
git submodule init
git submodule update
make run -j 4
```