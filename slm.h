#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"

typedef struct {
    SSM* ssm;                   // Reuse SSM implementation
    float* d_softmax;           // seq_len x batch_size x output_dim (probabilities)
    float* d_log_probs;         // seq_len x batch_size (log probabilities for targets)
    float* d_loss_buffer;       // Temporary buffer for loss calculation
} SLM;

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

// CUDA kernel for cross-entropy loss calculation
__global__ void cross_entropy_loss_kernel_slm(float* log_probs, float* softmax_probs, float* targets, 
                                              int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float* softmax_ptr = softmax_probs + batch_idx * vocab_size;
    float* targets_ptr = targets + batch_idx * vocab_size;
    
    // Find target index
    float log_prob = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (targets_ptr[i] > 0.5f) {
            float prob = fmaxf(softmax_ptr[i], 1e-15f); // Avoid log(0)
            log_prob = -logf(prob);
            break;
        }
    }
    
    log_probs[batch_idx] = log_prob;
}

// CUDA kernel for cross-entropy gradient
__global__ void cross_entropy_gradient_kernel_slm(float* grad, float* predictions, float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = predictions[idx] - targets[idx];
    }
}

// Initialize the state space language model
SLM* init_slm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Initialize underlying SSM
    slm->ssm = init_ssm(input_dim, state_dim, output_dim, seq_len, batch_size);
    
    // Allocate additional device memory for language modeling
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_log_probs, seq_len * batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_loss_buffer, seq_len * batch_size * sizeof(float)));
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
    if (slm) {
        if (slm->ssm) free_ssm(slm->ssm);  
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_log_probs);
        cudaFree(slm->d_loss_buffer);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, float* d_X) {
    // Use SSM forward pass to get logits
    forward_pass_ssm(slm->ssm, d_X);
    
    // Apply softmax to each timestep
    for (int t = 0; t < slm->ssm->seq_len; t++) {
        float* logits_t = slm->ssm->d_predictions + t * slm->ssm->batch_size * slm->ssm->output_dim;
        float* softmax_t = slm->d_softmax + t * slm->ssm->batch_size * slm->ssm->output_dim;
        
        softmax_kernel_slm<<<slm->ssm->batch_size, 1>>>(softmax_t, logits_t, slm->ssm->batch_size, slm->ssm->output_dim);
    }
}

// Calculate cross-entropy loss
float calculate_loss_slm(SLM* slm, float* d_y) {
    // Calculate log probabilities for each timestep
    for (int t = 0; t < slm->ssm->seq_len; t++) {
        float* softmax_t = slm->d_softmax + t * slm->ssm->batch_size * slm->ssm->output_dim;
        float* targets_t = d_y + t * slm->ssm->batch_size * slm->ssm->output_dim;
        float* log_probs_t = slm->d_log_probs + t * slm->ssm->batch_size;
        
        cross_entropy_loss_kernel_slm<<<slm->ssm->batch_size, 1>>>(
            log_probs_t, softmax_t, targets_t, slm->ssm->batch_size, slm->ssm->output_dim
        );
    }
    
    // Sum all log probabilities using cuBLAS
    float total_loss;
    int total_elements = slm->ssm->seq_len * slm->ssm->batch_size;
    CHECK_CUBLAS(cublasSasum(slm->ssm->cublas_handle, total_elements, slm->d_log_probs, 1, &total_loss));
    
    // Calculate gradients for backward pass (softmax - targets)
    int total_output_elements = slm->ssm->seq_len * slm->ssm->batch_size * slm->ssm->output_dim;
    int block_size = 256;
    int num_blocks = (total_output_elements + block_size - 1) / block_size;
    
    cross_entropy_gradient_kernel_slm<<<num_blocks, block_size>>>(
        slm->ssm->d_error,
        slm->d_softmax,
        d_y,
        total_output_elements
    );
    
    return total_loss / total_elements;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_ssm(slm->ssm);
}

// Backward pass
void backward_pass_slm(SLM* slm, float* d_X) {
    backward_pass_ssm(slm->ssm, d_X);
}

// Update weights
void update_weights_slm(SLM* slm, float learning_rate) {
    update_weights_ssm(slm->ssm, learning_rate);
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    save_ssm(slm->ssm, filename);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    // First load the SSM
    SSM* ssm = load_ssm(filename, custom_batch_size);
    if (!ssm) return NULL;
    
    // Create SLM wrapper
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->ssm = ssm;
    
    // Allocate additional device memory for language modeling
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, ssm->seq_len * ssm->batch_size * ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_log_probs, ssm->seq_len * ssm->batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_loss_buffer, ssm->seq_len * ssm->batch_size * sizeof(float)));
    
    return slm;
}

#endif