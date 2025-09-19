# slm
A small language model implementation

Consider a small language model operating on sequences of character tokens. The architecture consists of token embeddings, sinusoidal positional encodings, a transformer backbone with causal self-attention, and an output projection layer. The token embedding layer maps discrete character tokens to continuous vector representations, while sinusoidal position encodings provide positional information. The transformer processes sequences through multiple layers of causal self-attention and feed-forward networks with residual connections. Each transformer layer applies attention followed by an MLP with swish activation. The output projection maps the final transformer layer to vocabulary logits for next-token prediction.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```
