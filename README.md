# slm
A small language model implementation

Consider a character-level language model built on a dual state space model backbone with MLP projection, operating on character sequences of shape (seq_len × batch_size). The architecture combines learned character embeddings with temporal state dynamics through two sequential SSM layers, followed by MLP transformation and softmax normalization for next-character prediction. The forward propagation follows:

$$
\begin{align*}
E_t &= W_E[X_t] \\
H^1_t &= E_tB_1^T + H^1_{t-1}A_1^T \\
O^1_t &= H^1_t\sigma(H^1_t) \\
Y^1_t &= O^1_tC_1^T + E_tD_1^T \\
H^2_t &= Y^1_tB_2^T + H^2_{t-1}A_2^T \\
O^2_t &= H^2_t\sigma(H^2_t) \\
Y^2_t &= O^2_tC_2^T + Y^1_tD_2^T \\
Z_t &= Y^2_tW_1 \\
A_t &= Z_t\sigma(Z_t) \\
L_t &= A_tW_2 \\
P_t &= \frac{\exp(L_t)}{\sum_c \exp(L_{t,c})}
\end{align*}
$$

The embedding matrix $W_E$ maps discrete character indices to dense vector representations via indexing $W_E[X_t]$. The first SSM processes embedded inputs through state transition matrix $A_1$, input matrix $B_1$, output matrix $C_1$, and feedthrough matrix $D_1$. The second SSM further processes the first SSM's output through its own parameters $A_2$, $B_2$, $C_2$, and $D_2$. The MLP then transforms the final SSM outputs through weight matrices $W_1$ and $W_2$ with Swish activation $z\sigma(z)$. Finally, softmax normalization produces probability distributions over the character vocabulary.

For language modeling, the cross-entropy loss between predicted and actual next characters is minimized, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\mathcal{L} &= -\frac{1}{T \cdot B}\sum_{t=1}^{T}\sum_{b=1}^{B} \log P_{t,b,y_{t,b}} \\
\frac{\partial \mathcal{L}}{\partial L_t} &= P_t - \mathbf{1}_{y_t} \\
\frac{\partial \mathcal{L}}{\partial W_2} &= A_t^T(\frac{\partial \mathcal{L}}{\partial L_t}) \\
\frac{\partial \mathcal{L}}{\partial A_t} &= (\frac{\partial \mathcal{L}}{\partial L_t})(W_2)^T \\
\frac{\partial \mathcal{L}}{\partial Z_t} &= \frac{\partial \mathcal{L}}{\partial A_t} \odot [\sigma(Z_t) + Z_t\sigma(Z_t)(1-\sigma(Z_t))] \\
\frac{\partial \mathcal{L}}{\partial W_1} &= Y_t^T(\frac{\partial \mathcal{L}}{\partial Z_t}) \\
\frac{\partial \mathcal{L}}{\partial Y_t} &= (\frac{\partial \mathcal{L}}{\partial Z_t})(W_1)^T
\end{align*}
$$

The gradient then flows backward through both SSM layers following standard BPTT with Swish derivatives. The gradient flows from the MLP back through the second SSM, then through the first SSM to the embeddings:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial C} &= \sum_t (\frac{\partial \mathcal{L}}{\partial Y_t})^T O_t \\
\frac{\partial \mathcal{L}}{\partial D} &= \sum_t (\frac{\partial \mathcal{L}}{\partial Y_t})^T E_t \\
\frac{\partial \mathcal{L}}{\partial O_t} &= (\frac{\partial \mathcal{L}}{\partial Y_t})C \\
\frac{\partial \mathcal{L}}{\partial H_t} &= \frac{\partial \mathcal{L}}{\partial O_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial \mathcal{L}}{\partial H_{t+1}})A \\
\frac{\partial \mathcal{L}}{\partial A} &= \sum_t (\frac{\partial \mathcal{L}}{\partial H_t})^T H_{t-1} \\
\frac{\partial \mathcal{L}}{\partial B} &= \sum_t (\frac{\partial \mathcal{L}}{\partial H_t})^T E_t \\
\frac{\partial \mathcal{L}}{\partial W_E[c]} &= \sum_{\substack{t,b \\ X_{t,b}=c}} \left(B^T\frac{\partial \mathcal{L}}{\partial H_t} + D^T\frac{\partial \mathcal{L}}{\partial Y_t}\right)
\end{align*}
$$

where $\mathbf{1}_{y_t}$ denotes the one-hot encoding of target characters. The embedding gradients accumulate contributions from all occurrences of each character across the sequence.

During generation, the model maintains hidden state across timesteps and samples from the predicted character distribution using temperature-controlled sampling:

$$
P_{\tau}(c) = \frac{\exp(L_c / \tau)}{\sum_{c'} \exp(L_{c'} / \tau)}
$$

where temperature $\tau$ controls sampling entropy - $\tau \rightarrow 0$ approaches argmax sampling while $\tau > 1$ increases randomness.

The AdamW optimizer maintains exponential moving averages for all parameters $\theta = \{A, B, C, D, W_E, W_1, W_2\}$ with momentum $\beta_1$, second moment $\beta_2$, and weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages CUDA for parallel computation across batch and sequence dimensions, with efficient kernel implementations for embedding lookup, Swish activation, softmax normalization, and gradient accumulation. Character sequences are extracted from text corpora through random position sampling, providing diverse training contexts.

## Architecture

This implementation uses a **dual-SSM architecture** with two sequential state space model layers:
- **Embedding Layer**: Maps characters to dense vectors
- **First SSM Layer**: Processes embedded inputs with temporal state dynamics
- **Second SSM Layer**: Further processes first SSM outputs for enhanced representation
- **MLP Layer**: Transforms final SSM outputs through two-layer MLP with Swish activation
- **Softmax Layer**: Produces probability distributions over character vocabulary

This dual-SSM design provides increased model capacity and representational power compared to single-SSM architectures.

## How to run
```
sudo apt update
sudo apt install clang time libcurl-dev
# Ensure CUDA toolkit is installed
make run
```