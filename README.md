# slm
A small language model implementation

Consider a small language model operating on sequences of character tokens. The architecture consists of token embeddings, sinusoidal positional encodings, a transformer backbone with causal self-attention, and an output projection layer. The forward propagation follows:

$$
\begin{align*}
E_{b,t,d} &= W_{emb}[x_{b,t}, d] \\
P_{b,t,d} &= E_{b,t,d} + \begin{cases}
\sin(t / 10000^{2k/d_{model}}) & \text{if } d = 2k \\
\cos(t / 10000^{2k/d_{model}}) & \text{if } d = 2k+1
\end{cases} \\
Q &= PW_q, \quad K = PW_k, \quad V = PW_v \\
S &= \frac{QK^T}{\sqrt{d}} \\
A_{ij} &= \begin{cases}
\frac{\exp(S_{ij})}{\sum_{k \leq j} \exp(S_{ik})} & \text{if } i \leq j \\
0 & \text{if } i > j
\end{cases} \\
Z &= AV \\
Z' &= ZW_o + P \\
H &= Z'W_1 \\
S' &= H \odot \sigma(H) \\
Y_{attn} &= S'W_2 + Z' \\
\hat{y}_{b,t,v} &= \frac{\exp(Y_{out,b,t,v})}{\sum_k \exp(Y_{out,b,t,k})} \\
L &= -\frac{1}{BT} \sum_{b,t} \log \hat{y}_{b,t,y_{b,t}}
\end{align*}
$$

The token embedding layer maps discrete character tokens to continuous vector representations, while sinusoidal position encodings provide positional information. The transformer processes sequences through multiple layers of causal self-attention and feed-forward networks with residual connections. Each transformer layer applies attention followed by an MLP with swish activation. The output projection maps the final transformer layer to vocabulary logits for next-token prediction. The backward pass computes gradients for each component through the chain rule:

$$
\begin{align*}
\frac{\partial L}{\partial Y_{out}} &= \hat{y} - y_{true} \\
\frac{\partial L}{\partial W_2} &= {S'}^T\left(\frac{\partial L}{\partial Y_{attn}}\right) \\
\frac{\partial L}{\partial S'} &= \left(\frac{\partial L}{\partial Y_{attn}}\right)W_2^T \\
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial S'} \odot \left[\sigma(H) + H \odot \sigma(H) \odot (1-\sigma(H))\right] \\
\frac{\partial L}{\partial W_1} &= {Z'}^T\left(\frac{\partial L}{\partial H}\right) \\
\frac{\partial L}{\partial Z'} &= \left(\frac{\partial L}{\partial H}\right)W_1^T + \frac{\partial L}{\partial Y_{attn}} \\
\frac{\partial L}{\partial W_o} &= Z^T\left(\frac{\partial L}{\partial Z'}\right) \\
\frac{\partial L}{\partial Z} &= \left(\frac{\partial L}{\partial Z'}\right)W_o^T \\
\frac{\partial L}{\partial A} &= \left(\frac{\partial L}{\partial Z}\right)V^T \\
\frac{\partial L}{\partial V} &= A^T\left(\frac{\partial L}{\partial Z}\right) \\
\frac{\partial L}{\partial S} &= A \odot \left(\frac{\partial L}{\partial A} - \sum_j \frac{\partial L}{\partial A} \odot A\right) \odot M_{causal} \\
\frac{\partial L}{\partial Q} &= \frac{1}{\sqrt{d}}\left(\frac{\partial L}{\partial S}\right)K \\
\frac{\partial L}{\partial K} &= \frac{1}{\sqrt{d}}\left(\frac{\partial L}{\partial S}\right)^TQ \\
\frac{\partial L}{\partial W_q} &= P^T\left(\frac{\partial L}{\partial Q}\right) \\
\frac{\partial L}{\partial W_k} &= P^T\left(\frac{\partial L}{\partial K}\right) \\
\frac{\partial L}{\partial W_v} &= P^T\left(\frac{\partial L}{\partial V}\right) \\
\frac{\partial L}{\partial P} &= \left(\frac{\partial L}{\partial Q}\right)W_q^T + \left(\frac{\partial L}{\partial K}\right)W_k^T + \left(\frac{\partial L}{\partial V}\right)W_v^T + \frac{\partial L}{\partial Z'} \\
\frac{\partial L}{\partial W_{emb}} &= \sum_{b,t} \frac{\partial L}{\partial E_{b,t}} \cdot \delta(x_{b,t})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```