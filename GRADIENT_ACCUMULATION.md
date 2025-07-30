# Gradient Accumulation in SLM

## Overview

This implementation includes gradient accumulation, which allows training with larger effective batch sizes than what might fit in GPU memory. This is particularly useful for training larger models or when working with limited memory resources.

## How It Works

Gradient accumulation works by:

1. **Processing Multiple Mini-batches**: Instead of updating weights after each batch, the model processes multiple smaller mini-batches
2. **Accumulating Gradients**: Gradients from each mini-batch are accumulated (summed) rather than immediately applied
3. **Scaling and Updating**: After processing all mini-batches in the accumulation cycle, gradients are scaled by `1/accumulation_steps` and weights are updated

## Mathematical Equivalence

Training with gradient accumulation over N steps is mathematically equivalent to training with a batch size that is N times larger, but uses only 1/N of the memory.

For example:
- Original: batch_size=256 (high memory usage)
- With accumulation: batch_size=64, accumulation_steps=4 (same effective batch size, 1/4 memory usage per mini-batch)

## Configuration

### Method 1: Command Line (Recommended)

```bash
# Standard training (no accumulation)
./train.out model.bin 1

# Gradient accumulation with 4 steps
./train.out model.bin 4

# Start new training with 8 accumulation steps
./train.out "" 8
```

### Method 2: Code Modification

In `train.c`, adjust the default value:

```c
int grad_accumulation_steps = 4;  // Default value if not specified via command line
```

## Benefits

1. **Memory Efficiency**: Allows larger effective batch sizes without proportional memory increase
2. **Training Stability**: Larger effective batch sizes can lead to more stable training
3. **Flexibility**: Can be easily disabled by setting `grad_accumulation_steps = 1`

## Implementation Details

### Files Modified

- **train.c**: Main training loop modified to handle gradient accumulation
- **slm.h**: Added `scale_gradients_slm()` function declaration and CUDA kernel prototype
- **slm.c**: Added gradient scaling implementation with CUDA kernel

### Key Functions

- `scale_gradients_slm()`: Scales embedding gradients by accumulation factor
- `scale_gradients_kernel()`: CUDA kernel for efficient gradient scaling

### Training Flow

```
For each accumulation cycle:
1. Zero gradients (once at start)
2. For each mini-batch in cycle:
   - Forward pass
   - Calculate loss
   - Backward pass (accumulate gradients)
3. Scale gradients by 1/accumulation_steps
4. Update weights
5. Reset accumulation counter
```

## Output

The training output now includes accumulation information:

```
Batch [42/100000], Loss: 2.1234, Avg Loss: 2.1456, LR: 0.000095, Accum: 3/4
```

Where:
- `Loss`: Current mini-batch loss
- `Avg Loss`: Average loss over current accumulation cycle
- `Accum`: Current accumulation progress (3 out of 4 mini-batches processed)

## Performance Considerations

- **Memory Usage**: Reduced peak memory usage during training
- **Computation**: Slight computational overhead for gradient scaling
- **Training Speed**: May be slightly slower due to additional operations, but enables training with larger effective batch sizes

## Example Usage

### Example 1: Standard Training
```bash
./train.out model.bin 1
```
- Batch size: 64
- Accumulation steps: 1 (no accumulation)
- Effective batch size: 64
- Memory usage: standard

### Example 2: Moderate Accumulation
```bash
./train.out model.bin 4
```
- Batch size: 64
- Accumulation steps: 4
- Effective batch size: 256
- Memory usage: same as standard, but 4× effective batch size

### Example 3: High Accumulation
```bash
./train.out model.bin 8
```
- Batch size: 64
- Accumulation steps: 8
- Effective batch size: 512
- Memory usage: same as standard, but 8× effective batch size