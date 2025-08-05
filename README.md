# slm
A small language model implementation

Consider a character-level language model built on a triple state space model backbone with per-layer MLP projections, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through three sequential SSM layers, each followed by its own MLP transformation, and final softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t]
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. Each SSM layer processes its inputs through the standard state space model formulation, followed by a layer-specific MLP:

$$
\begin{align*}
H_t^{(i)} &= X_t^{(i)}B^{(i)T} + H_{t-1}^{(i)}A^{(i)T} \\
O_t^{(i)} &= H_t^{(i)}\sigma(H_t^{(i)}) \\
Y_t^{(i)} &= O_t^{(i)}C^{(i)T} + X_t^{(i)}D^{(i)T} \\
Z_t^{(i)} &= Y_t^{(i)}W_1^{(i)} \\
A_t^{(i)} &= Z_t^{(i)}\sigma(Z_t^{(i)}) \\
X_t^{(i+1)} &= A_t^{(i)}W_2^{(i)}
\end{align*}
$$

where $i$ denotes the layer index. The state transition matrix $A^{(i)}$ captures temporal dependencies, input matrix $B^{(i)}$ maps current inputs to state updates, output matrix $C^{(i)}$ projects nonlinearly activated states to outputs, and feedthrough matrix $D^{(i)}$ provides direct input-output connections. Each layer's MLP transforms the SSM output with weights $W_1^{(i)}$ and $W_2^{(i)}$, using swish activation $z\sigma(z)$. The first SSM takes embeddings $E_t$ as input, and each subsequent SSM takes the previous layer's MLP output as input. The final layer's MLP produces vocabulary-sized outputs for character prediction:

$$
\begin{align*}
P &= \frac{\exp(X_t^{(L+1)})}{\sum_c \exp(X_{t,c}^{(L+1)})}
\end{align*}
$$

where $L$ is the number of layers.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
L &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}}
\end{align*}
$$

The gradient flows backward through each layer's MLP and SSM following the chain rule. For layer $i$:

$$
\begin{align*}
\frac{\partial L}{\partial L_t} &= P_t - \mathbf{1}_{y_t} \quad \text{(final layer only)} \\
\frac{\partial L}{\partial W_2^{(i)}} &= A_t^{(i)T}(\frac{\partial L}{\partial X_t^{(i+1)}}) \\
\frac{\partial L}{\partial A_t^{(i)}} &= (\frac{\partial L}{\partial X_t^{(i+1)}})(W_2^{(i)})^T \\
\frac{\partial L}{\partial Z_t^{(i)}} &= \frac{\partial L}{\partial A_t^{(i)}} \odot [\sigma(Z_t^{(i)}) + Z_t^{(i)}\sigma(Z_t^{(i)})(1-\sigma(Z_t^{(i)}))] \\
\frac{\partial L}{\partial W_1^{(i)}} &= Y_t^{(i)T}(\frac{\partial L}{\partial Z_t^{(i)}}) \\
\frac{\partial L}{\partial Y_t^{(i)}} &= (\frac{\partial L}{\partial Z_t^{(i)}})(W_1^{(i)})^T
\end{align*}
$$

The gradient then flows backward through each SSM layer following standard BPTT with Swish derivatives:

$$
\begin{align*}
\frac{\partial L}{\partial C^{(i)}} &= \sum_t (\frac{\partial L}{\partial Y_t^{(i)}})^T O_t^{(i)} \\
\frac{\partial L}{\partial D^{(i)}} &= \sum_t (\frac{\partial L}{\partial Y_t^{(i)}})^T X_t^{(i)} \\
\frac{\partial L}{\partial O_t^{(i)}} &= (\frac{\partial L}{\partial Y_t^{(i)}})C^{(i)} \\
\frac{\partial L}{\partial H_t^{(i)}} &= \frac{\partial L}{\partial O_t^{(i)}} \odot [\sigma(H_t^{(i)}) + H_t^{(i)}\sigma(H_t^{(i)})(1-\sigma(H_t^{(i)}))] + (\frac{\partial L}{\partial H_{t+1}^{(i)}})A^{(i)} \\
\frac{\partial L}{\partial A^{(i)}} &= \sum_t (\frac{\partial L}{\partial H_t^{(i)}})^T H_{t-1}^{(i)} \\
\frac{\partial L}{\partial B^{(i)}} &= \sum_t (\frac{\partial L}{\partial H_t^{(i)}})^T X_t^{(i)} \\
\frac{\partial L}{\partial W_E[c]} &= \sum_{\substack{t,b \\ X_{t,b}=c}} \left(B^{(0)T}\frac{\partial L}{\partial H_t^{(0)}} + D^{(0)T}\frac{\partial L}{\partial Y_t^{(0)}}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling combined with nucleus sampling:

$$
P_{\tau}(c) = \frac{\exp(L_c / \tau)}{\sum_{c'} \exp(L_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

Nucleus sampling further improves generation quality by restricting sampling to the nucleus of the probability distribution. For a given threshold $p \in (0,1)$, only tokens with cumulative probability $\leq p$ (when sorted in descending order by probability) are considered for sampling. This eliminates low-probability tokens that could lead to incoherent text, while preserving diversity within the most likely candidates.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A_1^{(i)}, B_1^{(i)}, C_1^{(i)}, D_1^{(i)}, W_1^{(i)}, W_2^{(i)}, ..., W_E\}$ across all layers $i$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$ in each layer, the update rule is:

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