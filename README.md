# slm
A small language model implementation

This project implements a character-level language model using a transformer architecture. The model processes sequences of individual characters rather than word tokens, learning to predict the next character given previous context.

The architecture begins with a token embedding layer that converts each character into a continuous vector representation. These embeddings are then augmented with sinusoidal positional encodings that provide the model with information about each character's position in the sequence.

The core of the model is a multi-layer transformer that processes the embedded sequences. Each transformer layer consists of two main components: a causal self-attention mechanism and a feed-forward network. The causal attention ensures that predictions for each position can only depend on previous positions, which is essential for autoregressive text generation. The attention mechanism computes query, key, and value projections, applies scaled dot-product attention with a causal mask, and projects the result back. The feed-forward network uses a swish activation function (a smooth, non-monotonic activation that multiplies the input by its sigmoid) and includes residual connections around both the attention and feed-forward components.

After processing through all transformer layers, an output projection layer maps the final hidden states to logits over the vocabulary (all 256 possible byte values). These logits are converted to probabilities using the softmax function, and the model is trained to maximize the probability of the correct next character using cross-entropy loss.

The training process uses the AdamW optimizer, which enhances the standard Adam optimizer by decoupling weight decay from the gradient-based update. AdamW maintains exponential moving averages of both gradients and squared gradients, using these to adapt the learning rate for each parameter individually. The weight decay acts as L2 regularization, encouraging the model to use smaller weights and improving generalization.

The implementation uses BLAS (Basic Linear Algebra Subprograms) for efficient matrix operations, allowing the model to train effectively on modern CPU hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```