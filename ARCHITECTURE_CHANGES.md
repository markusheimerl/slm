# SLM Architecture Refactoring

## Overview
Refactored the Small Language Model (SLM) architecture from SSM→SSM→MLP to an alternating SSM→MLP→SSM→MLP→SSM→MLP→SSM→MLP pattern.

## Previous Architecture
```
Input → Embedding → SSM1 → SSM2 → MLP → Softmax → Output
```
- 2 SSM layers (sequential)
- 1 MLP layer (final)
- Total: 3 processing layers

## New Architecture
```
Input → Embedding → SSM[0] → MLP[0] → SSM[1] → MLP[1] → SSM[2] → MLP[2] → SSM[3] → MLP[3] → Softmax → Output
```
- 4 SSM layers (alternating with MLPs)
- 4 MLP layers (alternating with SSMs)
- Total: 8 processing layers

## Key Changes Made

### 1. Data Structure Changes (`slm.h`)
- **Old**: `SSM* ssm1, *ssm2; MLP* mlp;`
- **New**: `SSM* ssm[4]; MLP* mlp[4];`
- Added `d_layer_outputs[8]` for intermediate layer outputs
- Added `d_layer_gradients[8]` for intermediate layer gradients
- Removed old buffers: `d_ssm1_output`, `d_ssm1_gradients`

### 2. Initialization (`init_slm`)
- Creates 4 SSM layers: all with dimensions `embed_dim → embed_dim`
- Creates 4 MLP layers:
  - First 3: `embed_dim → embed_dim` (intermediate layers)
  - Last 1: `embed_dim → vocab_size` (output layer)
- Allocates proper buffer sizes for all intermediate layers

### 3. Forward Pass (`forward_pass_slm`)
- Loops through 4 iterations of SSM→MLP processing
- Each SSM processes input through standard state space equations
- Each MLP processes SSM output through feedforward network
- Proper data flow: Embedding → SSM[0] → MLP[0] → SSM[1] → MLP[1] → SSM[2] → MLP[2] → SSM[3] → MLP[3] → Softmax

### 4. Backward Pass (`backward_pass_slm`)
- Reverses the forward flow for gradient computation
- Order: MLP[3] ← SSM[3] ← MLP[2] ← SSM[2] ← MLP[1] ← SSM[1] ← MLP[0] ← SSM[0] ← Embedding
- Properly computes gradients for each layer input using:
  - State path: `B^T * state_error`
  - Output path: `D^T * output_error`
- Handles different dimensions correctly (embed_dim vs vocab_size)

### 5. Weight Management
- **Zero Gradients**: Zeros gradients for all 4 SSM and 4 MLP layers
- **Update Weights**: Updates weights for all 4 SSM and 4 MLP layers using AdamW
- **Save Model**: Saves each layer to separate files (`*_ssm0.bin`, `*_mlp0.bin`, etc.)
- **Load Model**: Loads all layers from their respective files

### 6. Text Generation (`generate_text_slm`)
- Updated to process through all 8 layers during generation
- Maintains hidden states for all 4 SSM layers during generation
- Properly handles the alternating SSM/MLP processing pattern

## Mathematical Flow

### Forward Pass
```
E_t = W_E[X_t]                           // Embedding lookup

// Layer 0: SSM[0] → MLP[0]
H0_t = E_t B0^T + H0_{t-1} A0^T         // SSM state update
O0_t = H0_t σ(H0_t)                     // SSM activation
Y0_t = O0_t C0^T + E_t D0^T             // SSM output
Z0_t = Y0_t W0_1                        // MLP linear
A0_t = Z0_t σ(Z0_t)                     // MLP activation
M0_t = A0_t W0_2                        // MLP output

// Layer 1: SSM[1] → MLP[1]
H1_t = M0_t B1^T + H1_{t-1} A1^T        // SSM state update
O1_t = H1_t σ(H1_t)                     // SSM activation
Y1_t = O1_t C1^T + M0_t D1^T            // SSM output
Z1_t = Y1_t W1_1                        // MLP linear
A1_t = Z1_t σ(Z1_t)                     // MLP activation
M1_t = A1_t W1_2                        // MLP output

// Layer 2: SSM[2] → MLP[2]
H2_t = M1_t B2^T + H2_{t-1} A2^T        // SSM state update
O2_t = H2_t σ(H2_t)                     // SSM activation
Y2_t = O2_t C2^T + M1_t D2^T            // SSM output
Z2_t = Y2_t W2_1                        // MLP linear
A2_t = Z2_t σ(Z2_t)                     // MLP activation
M2_t = A2_t W2_2                        // MLP output

// Layer 3: SSM[3] → MLP[3] (Final)
H3_t = M2_t B3^T + H3_{t-1} A3^T        // SSM state update
O3_t = H3_t σ(H3_t)                     // SSM activation
Y3_t = O3_t C3^T + M2_t D3^T            // SSM output
Z3_t = Y3_t W3_1                        // MLP linear
A3_t = Z3_t σ(Z3_t)                     // MLP activation
L_t = A3_t W3_2                         // Final logits

P_t = softmax(L_t)                      // Probability distribution
```

### Backward Pass
Gradients flow in reverse order through all 8 layers, with proper accumulation for SSM gradients through both state and output paths.

## Benefits of New Architecture
1. **Increased Model Capacity**: 8 layers vs 3 layers
2. **Better Representational Power**: Alternating pattern allows for more complex feature learning
3. **Improved Gradient Flow**: More processing layers with direct connections
4. **Modular Design**: Each SSM-MLP pair can learn specialized representations

## Implementation Status
- ✅ All core functions updated
- ✅ Forward and backward passes implemented
- ✅ Weight management functions updated
- ✅ Save/load functionality updated
- ✅ Text generation updated
- ⚠️ Testing pending (requires CUDA environment)

The refactoring maintains all existing functionality while implementing the new alternating architecture pattern as requested.