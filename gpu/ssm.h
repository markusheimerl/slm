#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Error checking macros for CUDA and cuBLAS calls
// ---------------------------------------------------------------------
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// ---------------------------------------------------------------------
// Structure definition for the state-space model (SSM)
// This version uses cuBLAS and integrates the AdamW optimizer.
// Internally parameterizes A for stability.
// ---------------------------------------------------------------------
typedef struct {
    // State transition matrices (stored on device)
    float* d_A;  // state_dim x state_dim
    float* d_B;  // state_dim x input_dim
    float* d_C;  // output_dim x state_dim
    float* d_D;  // output_dim x input_dim

    // RMS Norm parameters
    float* d_rms_weight;   // state_dim
    float* d_rms_weight_grad; // state_dim
    float* d_rms_weight_m; // Adam state for RMS weight
    float* d_rms_weight_v; // Adam state for RMS weight
    float rms_eps;         // epsilon for numerical stability

    // Host copies for saving/loading the model
    float* h_A;
    float* h_B;
    float* h_C;
    float* h_D;
    float* h_rms_weight;

    // Gradients (device pointers)
    float* d_A_grad;
    float* d_B_grad;
    float* d_C_grad;
    float* d_D_grad;

    // Adam optimizer first (m) and second (v) moment estimates (device pointers)
    float* d_A_m;
    float* d_A_v;
    float* d_B_m;
    float* d_B_v;
    float* d_C_m;
    float* d_C_v;
    float* d_D_m;
    float* d_D_v;

    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter

    // Helper arrays (device pointers)
    float* d_state;         // batch_size x state_dim
    float* d_next_state;    // batch_size x state_dim
    float* d_pre_state;     // batch_size x state_dim (pre-activation state)
    float* d_norm_state;    // batch_size x state_dim (normalized state)
    float* d_predictions;   // batch_size x output_dim
    float* d_error;         // batch_size x output_dim
    float* d_state_error;   // batch_size x state_dim
    float* d_rms_denom;     // batch_size (denominator for RMS norm)
    float* d_rms_grad;      // batch_size x state_dim (gradient of RMS norm)

    // Temporary buffers for matrix operations
    float* d_temp_state;    // batch_size x state_dim
    float* d_temp_output;   // batch_size x output_dim
    float* d_A_stable;      // Internal stable version of A
    
    // CUDA library handles
    cublasHandle_t cublas_handle;

    // Dimensions of the network
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
} SSM;

// ---------------------------------------------------------------------
// CUDA kernel: Compute stable A matrix using tanh-based parameterization
// ---------------------------------------------------------------------
__global__ void compute_stable_A_kernel(float* A_stable, const float* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n*n) {
        int row = idx / n;
        int col = idx % n;
        
        if (row == col) {
            // Diagonal elements: scaled tanh for eigenvalue control
            A_stable[idx] = 0.9f * tanhf(A[idx]);
        } else {
            // Off-diagonal elements: scaled by matrix size
            A_stable[idx] = A[idx] / sqrtf((float)n);
        }
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Compute gradient for A analytically
// ---------------------------------------------------------------------
__global__ void compute_A_grad_from_stable_grad(float* A_grad, 
                                               const float* A_stable_grad, 
                                               const float* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n*n) {
        int row = idx / n;
        int col = idx % n;
        
        if (row == col) {
            // Diagonal derivative: d(tanh)/dA = sech²(A)
            float tanh_val = tanhf(A[idx]);
            float sech_squared = 1.0f - tanh_val * tanh_val;
            A_grad[idx] = A_stable_grad[idx] * 0.9f * sech_squared;
        } else {
            // Off-diagonal derivative: 1/sqrt(n)
            A_grad[idx] = A_stable_grad[idx] / sqrtf((float)n);
        }
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Calculate squared norms per row (for RMS normalization)
// ---------------------------------------------------------------------
__global__ void rms_norm_squared_kernel(float* rms_denom, const float* pre_state, 
                                        int batch_size, int state_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float squared_sum = 0.0f;
        for (int col = 0; col < state_dim; col++) {
            float val = pre_state[col * batch_size + row]; // Column-major
            squared_sum += val * val;
        }
        rms_denom[row] = squared_sum / state_dim;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Apply RMS normalization
// ---------------------------------------------------------------------
__global__ void rms_norm_kernel(float* norm_state, const float* pre_state, 
                                const float* rms_denom, const float* rms_weight,
                                float eps, int batch_size, int state_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * state_dim) {
        int col = idx / batch_size;
        int row = idx % batch_size;
        
        float denom = sqrtf(rms_denom[row] + eps);
        float val = pre_state[col * batch_size + row]; // Column-major
        norm_state[col * batch_size + row] = val * rms_weight[col] / denom;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Calculate gradients for RMS normalization
// ---------------------------------------------------------------------
__global__ void rms_norm_backward_kernel(float* pre_state_grad, float* rms_weight_grad,
                                        const float* d_state_error, const float* pre_state,
                                        const float* rms_denom, const float* rms_weight,
                                        float eps, int batch_size, int state_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * state_dim) {
        int col = idx / batch_size;
        int row = idx % batch_size;
        
        float val = pre_state[col * batch_size + row];
        float denom = sqrtf(rms_denom[row] + eps);
        float grad = d_state_error[col * batch_size + row];
        
        // Gradient for weights
        atomicAdd(&rms_weight_grad[col], grad * val / denom);
        
        // Gradient for pre_state
        float factor1 = rms_weight[col] / denom;
        
        // Contribution from the squared norm
        float factor2 = 0.0f;
        for (int k = 0; k < state_dim; k++) {
            float grad_k = d_state_error[k * batch_size + row];
            float val_k = pre_state[k * batch_size + row];
            factor2 += grad_k * val_k * rms_weight[k];
        }
        factor2 = -val * factor2 / (2.0f * state_dim * denom * denom * denom);
        
        pre_state_grad[col * batch_size + row] = grad * factor1 + factor2;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: swish activation forward pass
// swish(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------
__global__ void swish_forward_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: swish activation backward pass
// Computes derivative using: grad_output *= swish + sigmoid*(1-swish)
// ---------------------------------------------------------------------
__global__ void swish_backward_kernel(float* grad_output, const float* input, 
                                     const float* activated, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish = activated[idx];  // already computed activated value
        grad_output[idx] *= (swish + sigmoid * (1.0f - swish));
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Mean Squared Error loss computation (elementwise error)
// ---------------------------------------------------------------------
__global__ void mse_loss_kernel(float* error, const float* predictions, 
                               const float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        error[idx] = diff;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: AdamW update (per weight element)
// ---------------------------------------------------------------------
__global__ void adamw_update_kernel(float* W, const float* grad, float* m, float* v, 
                                   int size, float beta1, float beta2, float epsilon, 
                                   float weight_decay, float learning_rate, int batch_size, 
                                   float bias_correction1, float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// Function: init_ssm
// Initializes the SSM structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;

    // Set Adam hyperparameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->weight_decay = 0.01f;
    ssm->adam_t = 0;
    ssm->rms_eps = 1e-5f;

    // Create cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&ssm->cublas_handle));

    // Allocate host memory for weight matrices
    ssm->h_A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->h_B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->h_C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->h_D = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->h_rms_weight = (float*)malloc(state_dim * sizeof(float));

    // Initialize matrices with scaled random values
    float scale_A = 1.0f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);

    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->h_A[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_A;
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->h_B[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_B;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->h_C[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_C;
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->h_D[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_D;
    }
    // Initialize RMS Norm weights to 1.0
    for (int i = 0; i < state_dim; i++) {
        ssm->h_rms_weight[i] = 1.0f;
    }

    // Allocate device memory for weight matrices
    CHECK_CUDA(cudaMalloc(&ssm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_weight, state_dim * sizeof(float)));

    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&ssm->d_A_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_grad, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_weight_grad, state_dim * sizeof(float)));

    // Allocate device memory for Adam first and second moment estimates and initialize to zero
    CHECK_CUDA(cudaMalloc(&ssm->d_A_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_m, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_B_v, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_C_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_D_v, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_weight_m, state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_weight_v, state_dim * sizeof(float)));

    CHECK_CUDA(cudaMemset(ssm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_rms_weight_m, 0, state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(ssm->d_rms_weight_v, 0, state_dim * sizeof(float)));

    // Allocate helper arrays
    CHECK_CUDA(cudaMalloc(&ssm->d_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_next_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_pre_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_norm_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_state_error, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_denom, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_rms_grad, batch_size * state_dim * sizeof(float)));

    // Allocate temporary buffers
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_state, batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_temp_output, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ssm->d_A_stable, state_dim * state_dim * sizeof(float)));

    // Copy weight matrices from host to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A, ssm->h_A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, ssm->h_B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, ssm->h_C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, ssm->h_D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_rms_weight, ssm->h_rms_weight, state_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    return ssm;
}

// ---------------------------------------------------------------------
// Function: forward_pass
// Computes the forward pass:
//   Compute A_stable from A
//   pre_state = A_stable * state + B * X
//   norm_state = RMSNorm(pre_state)
//   next_state = swish(norm_state)
//   predictions = C * next_state + D * X
// Updates the internal state to next_state.
// ---------------------------------------------------------------------
void forward_pass(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // Compute stable A from A for this forward pass
    int n = ssm->state_dim;
    int block_size = 256;
    int num_blocks = (n * n + block_size - 1) / block_size;
    compute_stable_A_kernel<<<num_blocks, block_size>>>(ssm->d_A_stable, ssm->d_A, n);

    // Compute pre_state = A_stable * state
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->state_dim,
                            &alpha,
                            ssm->d_A_stable, ssm->state_dim,
                            ssm->d_state, ssm->state_dim,
                            &beta,
                            ssm->d_pre_state, ssm->state_dim));

    // Add input contribution: pre_state += B * X
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->input_dim,
                            &alpha,
                            ssm->d_B, ssm->state_dim,
                            d_X, ssm->input_dim,
                            &alpha, // Add to existing pre_state
                            ssm->d_pre_state, ssm->state_dim));

    // Apply RMS Normalization
    // Step 1: Calculate squared norms per row
    block_size = 256;
    num_blocks = (ssm->batch_size + block_size - 1) / block_size;
    rms_norm_squared_kernel<<<num_blocks, block_size>>>(
        ssm->d_rms_denom, ssm->d_pre_state, ssm->batch_size, ssm->state_dim);
    
    // Step 2: Apply normalization with weights
    block_size = 256;
    num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    rms_norm_kernel<<<num_blocks, block_size>>>(
        ssm->d_norm_state, ssm->d_pre_state, ssm->d_rms_denom, ssm->d_rms_weight, 
        ssm->rms_eps, ssm->batch_size, ssm->state_dim);

    // Apply swish activation: next_state = swish(norm_state)
    int total_state = ssm->batch_size * ssm->state_dim;
    block_size = 256;
    num_blocks = (total_state + block_size - 1) / block_size;
    swish_forward_kernel<<<num_blocks, block_size>>>(ssm->d_next_state, ssm->d_norm_state, total_state);
    
    // Compute predictions = C * next_state
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->state_dim,
                            &alpha,
                            ssm->d_C, ssm->output_dim,
                            ssm->d_next_state, ssm->state_dim,
                            &beta,
                            ssm->d_predictions, ssm->output_dim));
                             
    // Add direct feedthrough: predictions += D * X
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ssm->output_dim, ssm->batch_size, ssm->input_dim,
                            &alpha,
                            ssm->d_D, ssm->output_dim,
                            d_X, ssm->input_dim,
                            &alpha, // Add to existing predictions
                            ssm->d_predictions, ssm->output_dim));
    
    // Update internal state: state = next_state
    CHECK_CUDA(cudaMemcpy(ssm->d_state, ssm->d_next_state,
                         ssm->batch_size * ssm->state_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------------
// Function: calculate_loss
// Computes the Mean Squared Error loss between predictions and targets.
// ---------------------------------------------------------------------
float calculate_loss(SSM* ssm, float* d_y) {
    int size = ssm->batch_size * ssm->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mse_loss_kernel<<<num_blocks, block_size>>>(ssm->d_error, ssm->d_predictions, d_y, size);
    float loss = 0.0f;
    CHECK_CUBLAS(cublasSdot(ssm->cublas_handle, size,
                           ssm->d_error, 1,
                           ssm->d_error, 1,
                           &loss));
    return loss / size;
}

// ---------------------------------------------------------------------
// Function: zero_gradients
// Clears the gradient arrays on the device.
// ---------------------------------------------------------------------
void zero_gradients(SSM* ssm) {
    int size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    int size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    int size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    int size_D = ssm->output_dim * ssm->input_dim * sizeof(float);
    int size_rms = ssm->state_dim * sizeof(float);
    
    CHECK_CUDA(cudaMemset(ssm->d_A_grad, 0, size_A));
    CHECK_CUDA(cudaMemset(ssm->d_B_grad, 0, size_B));
    CHECK_CUDA(cudaMemset(ssm->d_C_grad, 0, size_C));
    CHECK_CUDA(cudaMemset(ssm->d_D_grad, 0, size_D));
    CHECK_CUDA(cudaMemset(ssm->d_rms_weight_grad, 0, size_rms));
}

// ---------------------------------------------------------------------
// Function: backward_pass
// Computes gradients through the network using the chain rule,
// including through the RMS normalization layer.
// ---------------------------------------------------------------------
void backward_pass(SSM* ssm, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f;

    // Gradient for C: d_C_grad = error * (next_state)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->output_dim, ssm->state_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_error, ssm->output_dim,
                            ssm->d_next_state, ssm->state_dim,
                            &beta,
                            ssm->d_C_grad, ssm->output_dim));
                             
    // Gradient for D: d_D_grad = error * (X)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->output_dim, ssm->input_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_error, ssm->output_dim,
                            d_X, ssm->input_dim,
                            &beta,
                            ssm->d_D_grad, ssm->output_dim));
                             
    // Compute state error: state_error = C^T * error
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            ssm->state_dim, ssm->batch_size, ssm->output_dim,
                            &alpha,
                            ssm->d_C, ssm->output_dim,
                            ssm->d_error, ssm->output_dim,
                            &beta,
                            ssm->d_state_error, ssm->state_dim));
                             
    // Apply swish backward: modify state_error in place
    int total_state = ssm->batch_size * ssm->state_dim;
    int block_size = 256;
    int num_blocks = (total_state + block_size - 1) / block_size;
    swish_backward_kernel<<<num_blocks, block_size>>>(ssm->d_state_error, 
                                                     ssm->d_norm_state, 
                                                     ssm->d_next_state, 
                                                     total_state);
    
    // Backpropagate through RMS normalization
    block_size = 256;
    num_blocks = (ssm->batch_size * ssm->state_dim + block_size - 1) / block_size;
    CHECK_CUDA(cudaMemset(ssm->d_rms_weight_grad, 0, ssm->state_dim * sizeof(float)));
    rms_norm_backward_kernel<<<num_blocks, block_size>>>(
        ssm->d_temp_state, // Temporary buffer to store pre_state gradient
        ssm->d_rms_weight_grad,
        ssm->d_state_error,
        ssm->d_pre_state,
        ssm->d_rms_denom,
        ssm->d_rms_weight,
        ssm->rms_eps,
        ssm->batch_size,
        ssm->state_dim);
    
    // Copy pre_state gradient for further backpropagation
    CHECK_CUDA(cudaMemcpy(ssm->d_state_error, ssm->d_temp_state,
                         ssm->batch_size * ssm->state_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
                                                      
    // First compute gradient for A_stable: temp = state_error * (state)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->state_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_state_error, ssm->state_dim,
                            ssm->d_state, ssm->state_dim,
                            &beta,
                            ssm->d_A_stable, ssm->state_dim));
                             
    // Convert A_stable gradient to A gradient
    int size_A = ssm->state_dim * ssm->state_dim;
    block_size = 256;
    num_blocks = (size_A + block_size - 1) / block_size;
    compute_A_grad_from_stable_grad<<<num_blocks, block_size>>>(
        ssm->d_A_grad, ssm->d_A_stable, ssm->d_A, ssm->state_dim);
                             
    // Gradient for B: d_B_grad = state_error * (X)^T
    CHECK_CUBLAS(cublasSgemm(ssm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            ssm->state_dim, ssm->input_dim, ssm->batch_size,
                            &alpha,
                            ssm->d_state_error, ssm->state_dim,
                            d_X, ssm->input_dim,
                            &beta,
                            ssm->d_B_grad, ssm->state_dim));
}

// ---------------------------------------------------------------------
// Function: update_weights
// Uses the AdamW optimizer to update each weight matrix
// ---------------------------------------------------------------------
void update_weights(SSM* ssm, float learning_rate) {
    ssm->adam_t++; // Increment time step
    float bias_correction1 = 1.0f - powf(ssm->beta1, (float)ssm->adam_t);
    float bias_correction2 = 1.0f - powf(ssm->beta2, (float)ssm->adam_t);

    int block_size = 256;
    int size_A = ssm->state_dim * ssm->state_dim;
    int size_B = ssm->state_dim * ssm->input_dim;
    int size_C = ssm->output_dim * ssm->state_dim;
    int size_D = ssm->output_dim * ssm->input_dim;
    int size_rms = ssm->state_dim;

    int num_blocks_A = (size_A + block_size - 1) / block_size;
    int num_blocks_B = (size_B + block_size - 1) / block_size;
    int num_blocks_C = (size_C + block_size - 1) / block_size;
    int num_blocks_D = (size_D + block_size - 1) / block_size;
    int num_blocks_rms = (size_rms + block_size - 1) / block_size;

    adamw_update_kernel<<<num_blocks_A, block_size>>>(
        ssm->d_A, ssm->d_A_grad, ssm->d_A_m, ssm->d_A_v,
        size_A, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_B, block_size>>>(
        ssm->d_B, ssm->d_B_grad, ssm->d_B_m, ssm->d_B_v,
        size_B, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_C, block_size>>>(
        ssm->d_C, ssm->d_C_grad, ssm->d_C_m, ssm->d_C_v,
        size_C, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update_kernel<<<num_blocks_D, block_size>>>(
        ssm->d_D, ssm->d_D_grad, ssm->d_D_m, ssm->d_D_v,
        size_D, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);
        
    adamw_update_kernel<<<num_blocks_rms, block_size>>>(
        ssm->d_rms_weight, ssm->d_rms_weight_grad, ssm->d_rms_weight_m, ssm->d_rms_weight_v,
        size_rms, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
        learning_rate, ssm->batch_size, bias_correction1, bias_correction2);
}

// ---------------------------------------------------------------------
// Function: free_ssm
// Frees all allocated memory (both device and host) and destroys the cuBLAS handle.
// ---------------------------------------------------------------------
void free_ssm(SSM* ssm) {
    // Free device memory
    cudaFree(ssm->d_A);
    cudaFree(ssm->d_B);
    cudaFree(ssm->d_C);
    cudaFree(ssm->d_D);
    cudaFree(ssm->d_rms_weight);
    cudaFree(ssm->d_A_grad);
    cudaFree(ssm->d_B_grad);
    cudaFree(ssm->d_C_grad);
    cudaFree(ssm->d_D_grad);
    cudaFree(ssm->d_rms_weight_grad);
    cudaFree(ssm->d_A_m);
    cudaFree(ssm->d_A_v);
    cudaFree(ssm->d_B_m);
    cudaFree(ssm->d_B_v);
    cudaFree(ssm->d_C_m);
    cudaFree(ssm->d_C_v);
    cudaFree(ssm->d_D_m);
    cudaFree(ssm->d_D_v);
    cudaFree(ssm->d_rms_weight_m);
    cudaFree(ssm->d_rms_weight_v);
    cudaFree(ssm->d_state);
    cudaFree(ssm->d_next_state);
    cudaFree(ssm->d_pre_state);
    cudaFree(ssm->d_norm_state);
    cudaFree(ssm->d_predictions);
    cudaFree(ssm->d_error);
    cudaFree(ssm->d_state_error);
    cudaFree(ssm->d_rms_denom);
    cudaFree(ssm->d_rms_grad);
    cudaFree(ssm->d_temp_state);
    cudaFree(ssm->d_temp_output);
    cudaFree(ssm->d_A_stable);

    // Free host memory
    free(ssm->h_A);
    free(ssm->h_B);
    free(ssm->h_C);
    free(ssm->h_D);
    free(ssm->h_rms_weight);

    // Destroy handle
    cublasDestroy(ssm->cublas_handle);

    free(ssm);
}

// ---------------------------------------------------------------------
// Function: save_model
// Saves the model weights to a binary file.
// ---------------------------------------------------------------------
void save_ssm(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }

    // Write dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Write Adam hyperparameters
    fwrite(&ssm->beta1, sizeof(float), 1, file);
    fwrite(&ssm->beta2, sizeof(float), 1, file);
    fwrite(&ssm->epsilon, sizeof(float), 1, file);
    fwrite(&ssm->weight_decay, sizeof(float), 1, file);
    fwrite(&ssm->adam_t, sizeof(int), 1, file);
    fwrite(&ssm->rms_eps, sizeof(float), 1, file);

    size_t size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    size_t size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    size_t size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    size_t size_D = ssm->output_dim * ssm->input_dim * sizeof(float);
    size_t size_rms = ssm->state_dim * sizeof(float);

    // Allocate temporary host buffers
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);
    float* h_rms_weight = (float*)malloc(size_rms);
    
    // Allocate host buffers for Adam state
    float* h_A_m = (float*)malloc(size_A);
    float* h_A_v = (float*)malloc(size_A);
    float* h_B_m = (float*)malloc(size_B);
    float* h_B_v = (float*)malloc(size_B);
    float* h_C_m = (float*)malloc(size_C);
    float* h_C_v = (float*)malloc(size_C);
    float* h_D_m = (float*)malloc(size_D);
    float* h_D_v = (float*)malloc(size_D);
    float* h_rms_weight_m = (float*)malloc(size_rms);
    float* h_rms_weight_v = (float*)malloc(size_rms);

    // Copy weight matrices from device to host
    CHECK_CUDA(cudaMemcpy(h_A, ssm->d_A, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B, ssm->d_B, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C, ssm->d_C, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D, ssm->d_D, size_D, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_rms_weight, ssm->d_rms_weight, size_rms, cudaMemcpyDeviceToHost));
    
    // Copy Adam state from device to host
    CHECK_CUDA(cudaMemcpy(h_A_m, ssm->d_A_m, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_A_v, ssm->d_A_v, size_A, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_m, ssm->d_B_m, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_B_v, ssm->d_B_v, size_B, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_m, ssm->d_C_m, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_v, ssm->d_C_v, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_m, ssm->d_D_m, size_D, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_D_v, ssm->d_D_v, size_D, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_rms_weight_m, ssm->d_rms_weight_m, size_rms, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_rms_weight_v, ssm->d_rms_weight_v, size_rms, cudaMemcpyDeviceToHost));

    // Write weight matrices to file
    fwrite(h_A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(h_rms_weight, sizeof(float), ssm->state_dim, file);
    
    // Write Adam state to file
    fwrite(h_A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(h_B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(h_C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(h_D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(h_D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(h_rms_weight_m, sizeof(float), ssm->state_dim, file);
    fwrite(h_rms_weight_v, sizeof(float), ssm->state_dim, file);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_rms_weight);
    free(h_A_m);
    free(h_A_v);
    free(h_B_m);
    free(h_B_v);
    free(h_C_m);
    free(h_C_v);
    free(h_D_m);
    free(h_D_v);
    free(h_rms_weight_m);
    free(h_rms_weight_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: load_model
// Loads the model weights from a binary file and initializes a new SSM.
// ---------------------------------------------------------------------
SSM* load_ssm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }

    int input_dim, state_dim, output_dim, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Read Adam hyperparameters
    fread(&ssm->beta1, sizeof(float), 1, file);
    fread(&ssm->beta2, sizeof(float), 1, file);
    fread(&ssm->epsilon, sizeof(float), 1, file);
    fread(&ssm->weight_decay, sizeof(float), 1, file);
    fread(&ssm->adam_t, sizeof(int), 1, file);
    fread(&ssm->rms_eps, sizeof(float), 1, file);

    size_t size_A = state_dim * state_dim * sizeof(float);
    size_t size_B = state_dim * input_dim * sizeof(float);
    size_t size_C = output_dim * state_dim * sizeof(float);
    size_t size_D = output_dim * input_dim * sizeof(float);
    size_t size_rms = state_dim * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);
    float* h_D = (float*)malloc(size_D);
    float* h_rms_weight = (float*)malloc(size_rms);
    
    // Allocate host buffers for Adam state
    float* h_A_m = (float*)malloc(size_A);
    float* h_A_v = (float*)malloc(size_A);
    float* h_B_m = (float*)malloc(size_B);
    float* h_B_v = (float*)malloc(size_B);
    float* h_C_m = (float*)malloc(size_C);
    float* h_C_v = (float*)malloc(size_C);
    float* h_D_m = (float*)malloc(size_D);
    float* h_D_v = (float*)malloc(size_D);
    float* h_rms_weight_m = (float*)malloc(size_rms);
    float* h_rms_weight_v = (float*)malloc(size_rms);

    fread(h_A, sizeof(float), state_dim * state_dim, file);
    fread(h_B, sizeof(float), state_dim * input_dim, file);
    fread(h_C, sizeof(float), output_dim * state_dim, file);
    fread(h_D, sizeof(float), output_dim * input_dim, file);
    fread(h_rms_weight, sizeof(float), state_dim, file);
    
    // Read Adam state from file
    fread(h_A_m, sizeof(float), state_dim * state_dim, file);
    fread(h_A_v, sizeof(float), state_dim * state_dim, file);
    fread(h_B_m, sizeof(float), state_dim * input_dim, file);
    fread(h_B_v, sizeof(float), state_dim * input_dim, file);
    fread(h_C_m, sizeof(float), output_dim * state_dim, file);
    fread(h_C_v, sizeof(float), output_dim * state_dim, file);
    fread(h_D_m, sizeof(float), output_dim * input_dim, file);
    fread(h_D_v, sizeof(float), output_dim * input_dim, file);
    fread(h_rms_weight_m, sizeof(float), state_dim, file);
    fread(h_rms_weight_v, sizeof(float), state_dim, file);

    CHECK_CUDA(cudaMemcpy(ssm->d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D, h_D, size_D, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_rms_weight, h_rms_weight, size_rms, cudaMemcpyHostToDevice));
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(ssm->d_A_m, h_A_m, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_A_v, h_A_v, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_m, h_B_m, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_B_v, h_B_v, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_m, h_C_m, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_C_v, h_C_v, size_C, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_m, h_D_m, size_D, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_D_v, h_D_v, size_D, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_rms_weight_m, h_rms_weight_m, size_rms, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ssm->d_rms_weight_v, h_rms_weight_v, size_rms, cudaMemcpyHostToDevice));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_rms_weight);
    free(h_A_m);
    free(h_A_v);
    free(h_B_m);
    free(h_B_v);
    free(h_C_m);
    free(h_C_v);
    free(h_D_m);
    free(h_D_v);
    free(h_rms_weight_m);
    free(h_rms_weight_v);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ssm;
}

#endif