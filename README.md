# slm
A state space language model implementation

Consider a sequence model operating on batched inputs of shape (batch_size × seq_length × embed_dim). The architecture processes sequences through alternating token-mixing and channel-mixing operations, where the forward propagation follows:

$$
\begin{align*}
T &= X^TW_{\text{t}} \\
X' &= (T\odot\sigma(T))^T + X \\
C &= X'W_{\text{c}} \\
Y &= (C\odot\sigma(C)) + X'
\end{align*}
$$

The weight matrix $W_{\text{t}}$ is constrained to be lower triangular, ensuring causal processing where each position only attends to previous positions. The SiLU activation $x\sigma(x)$ provides smooth nonlinearity, yielding the following backward pass through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial C} &= \frac{\partial L}{\partial Y} \odot [\sigma(C) + C\odot\sigma(C)\odot(1-\sigma(C))] \\
\frac{\partial L}{\partial W_{\text{c}}} &= {X'}^T\frac{\partial L}{\partial C} \\
\frac{\partial L}{\partial X'} &= \frac{\partial L}{\partial Y} + \frac{\partial L}{\partial C}W_{\text{c}}^T \\
\frac{\partial L}{\partial T} &= \frac{\partial L}{\partial X'}^T \odot [\sigma(T) + T\odot\sigma(T)\odot(1-\sigma(T))] \\
\frac{\partial L}{\partial W_{\text{t}}} &= X\frac{\partial L}{\partial T} \odot M \\
\frac{\partial L}{\partial X} &= (\frac{\partial L}{\partial T}W_{\text{t}}^T)^T + \frac{\partial L}{\partial X'}
\end{align*}
$$

where $M$ is a lower triangular mask and $\odot$ denotes element-wise multiplication.

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m_t &= \beta_1m_{t-1} + (1-\beta_1)g_t \\
v_t &= \beta_2v_{t-1} + (1-\beta_2)g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
W_t &= (1-\lambda\eta)W_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

The implementation leverages cuBLAS for matrix operations, enabling efficient computation on CUDA-capable hardware.