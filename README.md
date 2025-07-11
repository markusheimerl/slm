# slm
A small language model implementation

Consider a character-level language model built on a state space model backbone, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics, using softmax normalization and cross-entropy loss for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t] \\
H_t &= E_tB^T + H_{t-1}A^T \\
O_t &= H_t\sigma(H_t) \\
Y_t &= O_tC^T + E_tD^T \\
P_t &= \frac{\exp(Y_t)}{\sum_c \exp(Y_{t,c})}
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps embeddings to state updates, output matrix $C$ projects states to outputs, and feedthrough matrix $D$ provides direct embedding-output connections. The softmax normalization produces probability distributions over the character vocabulary.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized:

$$
\begin{align*}
\mathcal{L} &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}} \
\frac{\partial \mathcal{L}}{\partial Y_t} &= P_t - \mathbf{1}{y_t} \
\frac{\partial \mathcal{L}}{\partial W_E[c]} &= \sum{\substack{t,b \ X_{t,b}=c}} \left(B^T\frac{\partial \mathcal{L}}{\partial H_t} + D^T\frac{\partial \mathcal{L}}{\partial Y_t}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters $y_t$. The gradient computation extends standard BPTT with embedding-specific updates that accumulate contributions from all occurrences of each character.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling:

$$
P_{\tau}(c) = \frac{\exp(Y_c / \tau)}{\sum_{c'} \exp(Y_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A, B, C, D, W_E\}$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$:

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta \mathcal{L} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta \mathcal{L})^2 \\
\theta_t &= (1-\lambda\eta)\theta_{t-1} - \eta\cdot\frac{m_t}{1-\beta_1^t}/\sqrt{\frac{v_t}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages CUDA for parallel computation across batch and sequence dimensions, with efficient kernel implementations for embedding lookup, swish activation, softmax normalization, and gradient accumulation. Character sequences are extracted from text corpora through random position sampling, providing diverse training contexts.

## How to run
```
sudo apt update
sudo apt install clang time
# Ensure CUDA toolkit is installed
make run
```