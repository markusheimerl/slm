# slm
A small language model implementation

The Mixer Model implements a language model architecture based on the MLP-Mixer design principles, operating on text data with separate token and channel mixing components.

Consider a batch of sequences with shape $(B \times S \times E)$ where $B$ is batch size, $S$ is sequence length, and $E$ is embedding dimension. The architecture consists of embedding/projection layers surrounding multiple MixerBlocks, where each block has two primary components:

## Token Mixing

The token mixing operation acts on each embedding dimension independently across the sequence dimension:

$$
\begin{align*}
X_{\text{norm}} &= \text{LayerNorm}(X) \\
X_{\text{trans}} &= \text{Transpose}(X_{\text{norm}}) \quad [B \times E \times S] \\
Y_{\text{token}} &= X_{\text{trans}} \cdot W_{\text{token}}^T \\
Y_{\text{act}} &= \text{SiLU}(Y_{\text{token}}) \\
Y_{\text{output}} &= \text{Transpose}(Y_{\text{act}}) + X
\end{align*}
$$

Where $W_{\text{token}}$ is a learnable weight matrix of shape $[S \times S]$ with a lower triangular masking pattern to ensure that tokens only attend to previous positions, implementing causal attention.

## Channel Mixing

The channel mixing operation processes embedding dimensions independently for each position:

$$
\begin{align*}
Y_{\text{norm}} &= \text{LayerNorm}(Y_{\text{output}}) \\
Z_{\text{channel}} &= Y_{\text{norm}} \cdot W_{\text{channel}}^T \\
Z_{\text{act}} &= \text{SiLU}(Z_{\text{channel}}) \\
Z_{\text{output}} &= Z_{\text{act}} + Y_{\text{output}}
\end{align*}
$$

Where $W_{\text{channel}}$ is a learnable weight matrix of shape $[E \times E]$.

## Backward Pass

The backward pass follows the chain rule through the network, computing gradients for:

1. The channel mixing layer: $\frac{\partial L}{\partial W_{\text{channel}}} = \frac{\partial L}{\partial Z_{\text{act}}} \cdot \text{SiLU}'(Z_{\text{channel}}) \cdot Y_{\text{norm}}$
2. The token mixing layer: $\frac{\partial L}{\partial W_{\text{token}}} = \frac{\partial L}{\partial Y_{\text{act}}} \cdot \text{SiLU}'(Y_{\text{token}}) \cdot X_{\text{trans}}$
3. The layer normalization parameters

The SiLU derivative is given by $\text{SiLU}'(x) = \sigma(x) + x\sigma(x)(1-\sigma(x))$ where $\sigma$ is the sigmoid function.

## Optimization

The AdamW optimizer maintains first and second moments of gradients and applies weight decay separately:

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= (1-\lambda\eta)\theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

Where $\eta$ is the learning rate, $\lambda$ is the weight decay coefficient, and $\beta_1, \beta_2, \epsilon$ are Adam hyperparameters.

The implementation leverages cuBLAS for matrix operations, enabling efficient computation on CUDA-capable hardware.
