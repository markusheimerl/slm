# slm
A small language model implementation

Consider a character-level language model built on a triple state space model backbone with MLP projection, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through three sequential SSM layers, followed by MLP transformation and softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t]
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. Both SSM layers then process their inputs through the standard state space model formulation:

$$
\begin{align*}
H_t &= X_tB^T + H_{t-1}A^T \\
O_t &= H_t\sigma(H_t) \\
Y_t &= O_tC^T + X_tD^T
\end{align*}
$$

The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps current inputs to state updates, output matrix $C$ projects nonlinearly activated states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The first SSM takes embeddings $E_t$ as input $X_t$, the second SSM takes the first SSM's output as its input, and the third SSM takes the second SSM's output as its input. The MLP then transforms the final SSM outputs:

$$
\begin{align*}
Z &= YW_1 \\
A &= Z\sigma(Z) \\
L &= AW_2 \\
P &= \frac{\exp(L)}{\sum_c \exp(L_c)}
\end{align*}
$$

The swish activation $z\sigma(z)$ interpolates between linear and nonlinear regimes, followed by softmax normalization to produce probability distributions over the character vocabulary.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
L &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}}
\end{align*}
$$

The gradient flows backward through the MLP following the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial L_t} &= P_t - \mathbf{1}_{y_t} \\
\frac{\partial L}{\partial W_2} &= A_t^T(\frac{\partial L}{\partial L_t}) \\
\frac{\partial L}{\partial A_t} &= (\frac{\partial L}{\partial L_t})(W_2)^T \\
\frac{\partial L}{\partial Z_t} &= \frac{\partial L}{\partial A_t} \odot [\sigma(Z_t) + Z_t\sigma(Z_t)(1-\sigma(Z_t))] \\
\frac{\partial L}{\partial W_1} &= Y_t^T(\frac{\partial L}{\partial Z_t}) \\
\frac{\partial L}{\partial Y_t} &= (\frac{\partial L}{\partial Z_t})(W_1)^T
\end{align*}
$$

The gradient then flows backward through all three SSM layers following standard BPTT with Swish derivatives:

$$
\begin{align*}
\frac{\partial L}{\partial C} &= \sum_t (\frac{\partial L}{\partial Y_t})^T O_t \\
\frac{\partial L}{\partial D} &= \sum_t (\frac{\partial L}{\partial Y_t})^T X_t \\
\frac{\partial L}{\partial O_t} &= (\frac{\partial L}{\partial Y_t})C \\
\frac{\partial L}{\partial H_t} &= \frac{\partial L}{\partial O_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial L}{\partial H_{t+1}})A \\
\frac{\partial L}{\partial A} &= \sum_t (\frac{\partial L}{\partial H_t})^T H_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial H_t})^T X_t \\
\frac{\partial L}{\partial W_E[c]} &= \sum_{\substack{t,b \\ X_{t,b}=c}} \left(B^T\frac{\partial L}{\partial H_t} + D^T\frac{\partial L}{\partial Y_t}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling combined with Top-p (nucleus) sampling:

$$
P_{\tau}(c) = \frac{\exp(L_c / \tau)}{\sum_{c'} \exp(L_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

Top-p sampling further improves generation quality by restricting sampling to the nucleus of the probability distribution. For a given threshold $p \in (0,1)$, only tokens with cumulative probability $\leq p$ (when sorted in descending order by probability) are considered for sampling. This eliminates low-probability tokens that could lead to incoherent text, while preserving diversity within the most likely candidates.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A_1, B_1, C_1, D_1, ..., W_E, W_1, W_2\}$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

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