#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

// Initialize the state space language model
SLM* init_slm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->input_dim = input_dim;
    slm->state_dim = state_dim;
    slm->output_dim = output_dim;
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.001f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    
    // Allocate host memory for initialization
    float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize matrices with careful scaling for stability
    float scale_B = 0.5f / sqrt(input_dim);
    float scale_C = 0.5f / sqrt(state_dim);
    float scale_D = 0.1f / sqrt(input_dim);
    
    // Initialize A as a stable matrix with eigenvalues < 1
    for (int i = 0; i < state_dim * state_dim; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * 0.05f;
    }
    
    // Add diagonal stability
    for (int i = 0; i < state_dim; i++) {
        A[i * state_dim + i] = 0.5f + ((float)rand() / (float)RAND_MAX * 0.3f);
    }
    
    for (int i = 0; i < state_dim * input_dim; i++) {
        B[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_B;
    }
    
    for (int i = 0; i < output_dim * state_dim; i++) {
        C[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_C;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        D[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale_D;
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&slm->d_A, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_B, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_C, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_D, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&slm->d_A_grad, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_B_grad, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_C_grad, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_D_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&slm->d_A_m, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_A_v, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_B_m, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_B_v, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_C_m, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_C_v, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_D_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_D_v, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for helper arrays
    CHECK_CUDA(cudaMalloc(&slm->d_states, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_predictions, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_error, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_state_error, seq_len * batch_size * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_state_outputs, seq_len * batch_size * state_dim * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_A_m, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_A_v, 0, state_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_B_m, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_B_v, 0, state_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_C_m, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_C_v, 0, output_dim * state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_D_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_D_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    free(D);
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
    // Free device memory
    cudaFree(slm->d_A); cudaFree(slm->d_B); cudaFree(slm->d_C); cudaFree(slm->d_D);
    cudaFree(slm->d_A_grad); cudaFree(slm->d_B_grad); cudaFree(slm->d_C_grad); cudaFree(slm->d_D_grad);
    cudaFree(slm->d_A_m); cudaFree(slm->d_A_v); cudaFree(slm->d_B_m); cudaFree(slm->d_B_v);
    cudaFree(slm->d_C_m); cudaFree(slm->d_C_v); cudaFree(slm->d_D_m); cudaFree(slm->d_D_v);
    cudaFree(slm->d_states); cudaFree(slm->d_predictions); cudaFree(slm->d_softmax); 
    cudaFree(slm->d_error); cudaFree(slm->d_state_error); cudaFree(slm->d_state_outputs);
    
    // Destroy cuBLAS handle
    cublasDestroy(slm->cublas_handle);
    
    free(slm);
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_slm(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_slm(float* grad_input, float* grad_output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        grad_input[idx] = grad_output[idx] * sigmoid * (1.0f + x * (1.0f - sigmoid));
    }
}

// CUDA kernel for softmax (per sequence element)
__global__ void softmax_kernel_slm(float* output, float* input, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float* input_ptr = input + batch_idx * vocab_size;
    float* output_ptr = output + batch_idx * vocab_size;
    
    // Find max for numerical stability
    float max_val = input_ptr[0];
    for (int i = 1; i < vocab_size; i++) {
        max_val = fmaxf(max_val, input_ptr[i]);
    }
    
    // Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float exp_val = expf(input_ptr[i] - max_val);
        output_ptr[i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        output_ptr[i] /= sum_exp;
    }
}

// CUDA kernel for cross-entropy gradient
__global__ void cross_entropy_gradient_kernel_slm(float* grad, float* predictions, float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = predictions[idx] - targets[idx];
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Clear states
    CHECK_CUDA(cudaMemset(slm->d_states, 0, slm->seq_len * slm->batch_size * slm->state_dim * sizeof(float)));
    
    for (int t = 0; t < slm->seq_len; t++) {
        // Pointers to current timestep data
        float* d_X_t = d_X + t * slm->batch_size * slm->input_dim;
        float* d_h_t = slm->d_states + t * slm->batch_size * slm->state_dim;
        
        // H_t = X_t B^T
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                slm->state_dim,
                                slm->batch_size,
                                slm->input_dim,
                                &alpha,
                                slm->d_B,
                                slm->input_dim,
                                d_X_t,
                                slm->input_dim,
                                &beta,
                                d_h_t,
                                slm->state_dim));
        
        // H_t += H_{t-1} A^T
        if (t > 0) {
            float* d_h_prev = slm->d_states + (t-1) * slm->batch_size * slm->state_dim;
            CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    slm->state_dim,
                                    slm->batch_size,
                                    slm->state_dim,
                                    &alpha,
                                    slm->d_A,
                                    slm->state_dim,
                                    d_h_prev,
                                    slm->state_dim,
                                    &beta_add,
                                    d_h_t,
                                    slm->state_dim));
        }
        
        // O_t = H_t σ(H_t)
        float* d_o_t = slm->d_state_outputs + t * slm->batch_size * slm->state_dim;
        int block_size = 256;
        int num_blocks = (slm->batch_size * slm->state_dim + block_size - 1) / block_size;
        swish_forward_kernel_slm<<<num_blocks, block_size>>>(d_o_t, d_h_t, slm->batch_size * slm->state_dim);
        
        // Y_t = O_t C^T + X_t D^T (logits)
        float* d_y_t = slm->d_predictions + t * slm->batch_size * slm->output_dim;
        
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                slm->output_dim,
                                slm->batch_size,
                                slm->state_dim,
                                &alpha,
                                slm->d_C,
                                slm->state_dim,
                                d_o_t,
                                slm->state_dim,
                                &beta,
                                d_y_t,
                                slm->output_dim));
        
        // Y_t += X_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                slm->output_dim,
                                slm->batch_size,
                                slm->input_dim,
                                &alpha,
                                slm->d_D,
                                slm->input_dim,
                                d_X_t,
                                slm->input_dim,
                                &beta_add,
                                d_y_t,
                                slm->output_dim));
        
        // Apply softmax to get probabilities
        float* d_softmax_t = slm->d_softmax + t * slm->batch_size * slm->output_dim;
        softmax_kernel_slm<<<slm->batch_size, 1>>>(d_softmax_t, d_y_t, slm->batch_size, slm->output_dim);
    }
}

// Calculate cross-entropy loss
float calculate_loss_slm(SLM* slm, float* d_y) {
    int total_elements = slm->seq_len * slm->batch_size * slm->output_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    // Calculate gradients (softmax - targets)
    cross_entropy_gradient_kernel_slm<<<num_blocks, block_size>>>(
        slm->d_error,
        slm->d_softmax,
        d_y,
        total_elements
    );

    // Calculate loss on CPU (more accurate for cross-entropy)
    float* h_softmax = (float*)malloc(total_elements * sizeof(float));
    float* h_targets = (float*)malloc(total_elements * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_softmax, slm->d_softmax, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_targets, d_y, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    double loss = 0.0;
    int valid_samples = 0;
    
    for (int i = 0; i < total_elements; i += slm->output_dim) {
        for (int j = 0; j < slm->output_dim; j++) {
            if (h_targets[i + j] == 1.0f) {
                float prob = h_softmax[i + j];
                prob = fmaxf(prob, 1e-15f); // Avoid log(0)
                loss -= log(prob);
                valid_samples++;
                break;
            }
        }
    }
    
    free(h_softmax);
    free(h_targets);
    
    return (float)(loss / valid_samples);
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    CHECK_CUDA(cudaMemset(slm->d_A_grad, 0, slm->state_dim * slm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_B_grad, 0, slm->state_dim * slm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_C_grad, 0, slm->output_dim * slm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_D_grad, 0, slm->output_dim * slm->input_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Clear state errors
    CHECK_CUDA(cudaMemset(slm->d_state_error, 0, slm->seq_len * slm->batch_size * slm->state_dim * sizeof(float)));
    
    for (int t = slm->seq_len - 1; t >= 0; t--) {
        float* d_X_t = d_X + t * slm->batch_size * slm->input_dim;
        float* d_h_t = slm->d_states + t * slm->batch_size * slm->state_dim;
        float* d_o_t = slm->d_state_outputs + t * slm->batch_size * slm->state_dim;
        float* d_dy_t = slm->d_error + t * slm->batch_size * slm->output_dim;
        float* d_dh_t = slm->d_state_error + t * slm->batch_size * slm->state_dim;
        
        // ∂L/∂C += (∂L/∂Y_t)^T O_t
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                slm->state_dim,
                                slm->output_dim,
                                slm->batch_size,
                                &alpha,
                                d_o_t,
                                slm->state_dim,
                                d_dy_t,
                                slm->output_dim,
                                &beta_add,
                                slm->d_C_grad,
                                slm->state_dim));
        
        // ∂L/∂D += (∂L/∂Y_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                slm->input_dim,
                                slm->output_dim,
                                slm->batch_size,
                                &alpha,
                                d_X_t,
                                slm->input_dim,
                                d_dy_t,
                                slm->output_dim,
                                &beta_add,
                                slm->d_D_grad,
                                slm->input_dim));
        
        // ∂L/∂O_t = (∂L/∂Y_t)C
        float* d_do_t = d_o_t; // reuse buffer
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                slm->state_dim,
                                slm->batch_size,
                                slm->output_dim,
                                &alpha,
                                slm->d_C,
                                slm->state_dim,
                                d_dy_t,
                                slm->output_dim,
                                &beta,
                                d_do_t,
                                slm->state_dim));
        
        // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))]
        int block_size = 256;
        int num_blocks = (slm->batch_size * slm->state_dim + block_size - 1) / block_size;
        swish_backward_kernel_slm<<<num_blocks, block_size>>>(d_dh_t, d_do_t, d_h_t, slm->batch_size * slm->state_dim);
        
        // ∂L/∂H_t += (∂L/∂H_{t+1})A
        if (t < slm->seq_len - 1) {
            float* d_dh_next = slm->d_state_error + (t+1) * slm->batch_size * slm->state_dim;
            CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    slm->state_dim,
                                    slm->batch_size,
                                    slm->state_dim,
                                    &alpha,
                                    slm->d_A,
                                    slm->state_dim,
                                    d_dh_next,
                                    slm->state_dim,
                                    &beta_add,
                                    d_dh_t,
                                    slm->state_dim));
        }
        
        // ∂L/∂B += (∂L/∂H_t)^T X_t
        CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                slm->input_dim,
                                slm->state_dim,
                                slm->batch_size,
                                &alpha,
                                d_X_t,
                                slm->input_dim,
                                d_dh_t,
                                slm->state_dim,
                                &beta_add,
                                slm->d_B_grad,
                                slm->input_dim));
        
        // ∂L/∂A += (∂L/∂H_t)^T H_{t-1}
        if (t > 0) {
            float* d_h_prev = slm->d_states + (t-1) * slm->batch_size * slm->state_dim;
            CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_T,
                                    slm->state_dim,
                                    slm->state_dim,
                                    slm->batch_size,
                                    &alpha,
                                    d_h_prev,
                                    slm->state_dim,
                                    d_dh_t,
                                    slm->state_dim,
                                    &beta_add,
                                    slm->d_A_grad,
                                    slm->state_dim));
        }
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_slm(
    float* weight,
    float* grad,
    float* m,
    float* v,
    float beta1,
    float beta2,
    float epsilon,
    float learning_rate,
    float weight_decay,
    float alpha_t,
    int size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)g
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)g²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·m̂/√v̂
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_slm(SLM* slm, float learning_rate) {
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update A
    int A_size = slm->state_dim * slm->state_dim;
    int A_blocks = (A_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<A_blocks, block_size>>>(
        slm->d_A, slm->d_A_grad, slm->d_A_m, slm->d_A_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, A_size, slm->batch_size
    );
    
    // Update B
    int B_size = slm->state_dim * slm->input_dim;
    int B_blocks = (B_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<B_blocks, block_size>>>(
        slm->d_B, slm->d_B_grad, slm->d_B_m, slm->d_B_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, B_size, slm->batch_size
    );
    
    // Update C
    int C_size = slm->output_dim * slm->state_dim;
    int C_blocks = (C_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<C_blocks, block_size>>>(
        slm->d_C, slm->d_C_grad, slm->d_C_m, slm->d_C_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, C_size, slm->batch_size
    );
    
    // Update D
    int D_size = slm->output_dim * slm->input_dim;
    int D_blocks = (D_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<D_blocks, block_size>>>(
        slm->d_D, slm->d_D_grad, slm->d_D_m, slm->d_D_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, D_size, slm->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Allocate temporary host memory
    float* A = (float*)malloc(slm->state_dim * slm->state_dim * sizeof(float));
    float* B = (float*)malloc(slm->state_dim * slm->input_dim * sizeof(float));
    float* C = (float*)malloc(slm->output_dim * slm->state_dim * sizeof(float));
    float* D = (float*)malloc(slm->output_dim * slm->input_dim * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(A, slm->d_A, slm->state_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B, slm->d_B, slm->state_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C, slm->d_C, slm->output_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D, slm->d_D, slm->output_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(A); free(B); free(C); free(D);
        return;
    }
    
    // Save dimensions
    fwrite(&slm->input_dim, sizeof(int), 1, file);
    fwrite(&slm->state_dim, sizeof(int), 1, file);
    fwrite(&slm->output_dim, sizeof(int), 1, file);
    fwrite(&slm->seq_len, sizeof(int), 1, file);
    fwrite(&slm->batch_size, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(A, sizeof(float), slm->state_dim * slm->state_dim, file);
    fwrite(B, sizeof(float), slm->state_dim * slm->input_dim, file);
    fwrite(C, sizeof(float), slm->output_dim * slm->state_dim, file);
    fwrite(D, sizeof(float), slm->output_dim * slm->input_dim, file);
    
    fwrite(&slm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* A_m = (float*)malloc(slm->state_dim * slm->state_dim * sizeof(float));
    float* A_v = (float*)malloc(slm->state_dim * slm->state_dim * sizeof(float));
    float* B_m = (float*)malloc(slm->state_dim * slm->input_dim * sizeof(float));
    float* B_v = (float*)malloc(slm->state_dim * slm->input_dim * sizeof(float));
    float* C_m = (float*)malloc(slm->output_dim * slm->state_dim * sizeof(float));
    float* C_v = (float*)malloc(slm->output_dim * slm->state_dim * sizeof(float));
    float* D_m = (float*)malloc(slm->output_dim * slm->input_dim * sizeof(float));
    float* D_v = (float*)malloc(slm->output_dim * slm->input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(A_m, slm->d_A_m, slm->state_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(A_v, slm->d_A_v, slm->state_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_m, slm->d_B_m, slm->state_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B_v, slm->d_B_v, slm->state_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_m, slm->d_C_m, slm->output_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_v, slm->d_C_v, slm->output_dim * slm->state_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_m, slm->d_D_m, slm->output_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(D_v, slm->d_D_v, slm->output_dim * slm->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(A_m, sizeof(float), slm->state_dim * slm->state_dim, file);
    fwrite(A_v, sizeof(float), slm->state_dim * slm->state_dim, file);
    fwrite(B_m, sizeof(float), slm->state_dim * slm->input_dim, file);
    fwrite(B_v, sizeof(float), slm->state_dim * slm->input_dim, file);
    fwrite(C_m, sizeof(float), slm->output_dim * slm->state_dim, file);
    fwrite(C_v, sizeof(float), slm->output_dim * slm->state_dim, file);
    fwrite(D_m, sizeof(float), slm->output_dim * slm->input_dim, file);
    fwrite(D_v, sizeof(float), slm->output_dim * slm->input_dim, file);
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v); free(B_m); free(B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, state_dim, output_dim, seq_len, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SLM* slm = init_slm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Allocate temporary host memory
    float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Load matrices
    fread(A, sizeof(float), state_dim * state_dim, file);
    fread(B, sizeof(float), state_dim * input_dim, file);
    fread(C, sizeof(float), output_dim * state_dim, file);
    fread(D, sizeof(float), output_dim * input_dim, file);
    
    fread(&slm->t, sizeof(int), 1, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* A_m = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* A_v = (float*)malloc(state_dim * state_dim * sizeof(float));
    float* B_m = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* B_v = (float*)malloc(state_dim * input_dim * sizeof(float));
    float* C_m = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* C_v = (float*)malloc(output_dim * state_dim * sizeof(float));
    float* D_m = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* D_v = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(A_m, sizeof(float), state_dim * state_dim, file);
    fread(A_v, sizeof(float), state_dim * state_dim, file);
    fread(B_m, sizeof(float), state_dim * input_dim, file);
    fread(B_v, sizeof(float), state_dim * input_dim, file);
    fread(C_m, sizeof(float), output_dim * state_dim, file);
    fread(C_v, sizeof(float), output_dim * state_dim, file);
    fread(D_m, sizeof(float), output_dim * input_dim, file);
    fread(D_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(slm->d_A_m, A_m, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_A_v, A_v, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_B_m, B_m, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_B_v, B_v, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_C_m, C_m, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_C_v, C_v, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_D_m, D_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_D_v, D_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(A); free(B); free(C); free(D);
    free(A_m); free(A_v); free(B_m); free(B_v);
    free(C_m); free(C_v); free(D_m); free(D_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return slm;
}

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_data_for_batch_processing(float* X, float* y, 
                                     float** X_reshaped, float** y_reshaped,
                                     int num_sequences, int seq_len, 
                                     int input_dim, int output_dim) {
    // Reshape to: seq_len tensors of size (batch_size x input_dim/output_dim)
    *X_reshaped = (float*)malloc(seq_len * num_sequences * input_dim * sizeof(float));
    *y_reshaped = (float*)malloc(seq_len * num_sequences * output_dim * sizeof(float));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time][feature]
            int orig_x_idx = b * seq_len * input_dim + t * input_dim;
            int orig_y_idx = b * seq_len * output_dim + t * output_dim;
            
            // New layout: [time][seq][feature] 
            int new_x_idx = t * num_sequences * input_dim + b * input_dim;
            int new_y_idx = t * num_sequences * output_dim + b * output_dim;
            
            memcpy(&(*X_reshaped)[new_x_idx], &X[orig_x_idx], input_dim * sizeof(float));
            memcpy(&(*y_reshaped)[new_y_idx], &y[orig_y_idx], output_dim * sizeof(float));
        }
    }
}

int main() {
    srand(time(NULL));
    
    // Parameters matching the new specifications
    const int input_dim = 512;      // EMBED_DIM - embedding dimension
    const int state_dim = 1024;     // Hidden state dimension
    const int seq_len = 1024;       // Sequence length
    const int num_sequences = 64;   // Batch size (number of sequences)
    const int output_dim = 256;     // MAX_CHAR_VALUE - vocabulary size for one-hot
    
    printf("Generating text sequence data...\n");
    printf("Input dim (embedding): %d\n", input_dim);
    printf("State dim (hidden): %d\n", state_dim);
    printf("Output dim (vocab): %d\n", output_dim);
    printf("Sequence length: %d\n", seq_len);
    printf("Number of sequences: %d\n", num_sequences);
    
    // Generate text sequence data from corpus
    float *X, *y;
    generate_text_sequence_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, 
                               "combined_corpus.txt");
    
    // Reshape data for batch processing (same as SSM)
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped,
                                    num_sequences, seq_len, input_dim, output_dim);
    
    // Allocate device memory for input and output and copy data
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_len * num_sequences * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_len * num_sequences * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X_reshaped, seq_len * num_sequences * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y_reshaped, seq_len * num_sequences * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize state space language model
    SLM* slm = init_slm(input_dim, state_dim, output_dim, seq_len, num_sequences);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    
    printf("\nStarting training...\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_slm(slm, d_X);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_y);

        // Print progress
        if (epoch > 0 && epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_X);
        
        // Update weights
        update_weights_slm(slm, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_slm_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_slm(slm, model_fname);
    save_text_sequence_data_to_csv(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    printf("\nTraining complete!\n");
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    cudaFree(d_X);
    cudaFree(d_y);
    free_slm(slm);
    
    return 0;
}