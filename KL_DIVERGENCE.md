# KL Divergence Penalty Implementation

This implementation adds a KL divergence penalty to the cross-entropy loss to regularize the language model's output distribution toward a uniform distribution.

## Mathematical Formulation

The total loss combines cross-entropy with KL divergence penalty:

**Total Loss:**
```
L_total = L_CE + λ_KL * KL(P||U)
```

Where:
- `L_CE` is the standard cross-entropy loss: `-log(P_{y_t})`
- `KL(P||U)` is the KL divergence between predicted distribution P and uniform distribution U
- `λ_KL` is the penalty weight (default: 0.01)

**KL Divergence:**
```
KL(P||U) = Σ P(c) log(P(c)/U(c))
         = Σ P(c) log(P(c)) - Σ P(c) log(1/V)
         = Σ P(c) log(P(c)) + log(V)
```

Since `log(V)` is constant, we compute the negative entropy: `-Σ P(c) log(P(c))`

**Gradient:**
The gradient with respect to logits includes both cross-entropy and KL divergence terms:
```
∂L_total/∂L_i = (P_i - 1_{y_i}) + λ_KL * P_i * (log(P_i) + 1)
```

## Implementation Details

### New Components Added:

1. **SLM Structure Updates:**
   - Added `float* d_kl_losses` buffer for KL divergence losses
   - Added `float kl_penalty_weight` parameter (default: 0.01)

2. **CUDA Kernels:**
   - `kl_divergence_loss_kernel`: Computes negative entropy `-Σ P(c) log(P(c))`
   - `kl_divergence_gradient_kernel`: Adds KL gradients to existing cross-entropy gradients

3. **Modified Functions:**
   - `calculate_loss_slm`: Now computes total loss = CE loss + KL penalty
   - `init_slm`/`load_slm`: Initialize KL penalty weight and allocate buffers
   - `save_slm`/`load_slm`: Persist KL penalty weight in model files

4. **New Function:**
   - `set_kl_penalty_weight_slm`: Dynamically adjust KL penalty weight

### Usage:

```c
// Initialize model (KL penalty weight defaults to 0.01)
SLM* slm = init_slm(embed_dim, state_dim, seq_len, batch_size);

// Optionally adjust KL penalty weight
set_kl_penalty_weight_slm(slm, 0.05f);  // Increase regularization
set_kl_penalty_weight_slm(slm, 0.0f);   // Disable KL penalty

// Training proceeds as normal - KL penalty is automatically included
```

## Effect on Training

The KL divergence penalty:
- **Regularizes** the model to avoid overly confident predictions
- **Promotes diversity** in output distributions  
- **Reduces overfitting** by preventing sharp probability distributions
- **Can improve generalization** especially with limited training data

Higher `λ_KL` values increase regularization strength but may hurt task performance if too large.