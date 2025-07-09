#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ssm/gpu/ssm.h"

#define VOCAB_SIZE 256

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    SSM* ssm;
    
    // Model parameters (on device)
    float* d_embed;      // 256 × embedding_dim
    float* d_proj;       // embedding_dim × 256
    
    // Gradients (on device)
    float* d_embed_grad; // 256 × embedding_dim
    float* d_proj_grad;  // embedding_dim × 256
    
    // Adam states
    float* d_embed_m, *d_embed_v;
    float* d_proj_m, *d_proj_v;
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Working buffers (on device)
    float* d_embedded;   // batch_size × embedding_dim
    float* d_logits;     // batch_size × 256
    float* d_logit_grad; // batch_size × 256
    float* d_loss;       // single loss value
    
    // Host buffers for data loading
    int* h_chars;        // context_length × batch_size
    int* h_next_chars;   // context_length × batch_size
    
    // Device buffers for characters
    int* d_chars;        // context_length × batch_size
    int* d_next_chars;   // context_length × batch_size
    
    // Hyperparameters
    int embedding_dim;
    int context_length;
    int batch_size;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} SLM;

// CUDA kernel for character embedding
__global__ void embed_chars_kernel(float* embedded, const float* embed_matrix, 
                                  const int* chars, int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * embed_dim) {
        int batch_idx = idx / embed_dim;
        int dim_idx = idx % embed_dim;
        int char_val = chars[batch_idx];
        embedded[idx] = embed_matrix[char_val * embed_dim + dim_idx];
    }
}

// CUDA kernel for cross-entropy loss and gradient computation
__global__ void cross_entropy_kernel(float* loss, float* grad, const float* logits,
                                    const int* targets, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        const float* logit_row = logits + batch_idx * VOCAB_SIZE;
        float* grad_row = grad + batch_idx * VOCAB_SIZE;
        int target = targets[batch_idx];
        
        // Find max for numerical stability
        float max_logit = logit_row[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            max_logit = fmaxf(max_logit, logit_row[i]);
        }
        
        // Compute softmax denominator
        float sum_exp = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            sum_exp += expf(logit_row[i] - max_logit);
        }
        
        // Loss for this sample
        float sample_loss = -logit_row[target] + max_logit + logf(sum_exp);
        atomicAdd(loss, sample_loss);
        
        // Gradient (softmax - one_hot)
        for (int i = 0; i < VOCAB_SIZE; i++) {
            float prob = expf(logit_row[i] - max_logit) / sum_exp;
            grad_row[i] = prob - (i == target ? 1.0f : 0.0f);
        }
    }
}

// CUDA kernel for embedding gradient accumulation
__global__ void accumulate_embed_grad_kernel(float* embed_grad, const float* output_grad,
                                           const int* chars, int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * embed_dim) {
        int batch_idx = idx / embed_dim;
        int dim_idx = idx % embed_dim;
        int char_val = chars[batch_idx];
        atomicAdd(&embed_grad[char_val * embed_dim + dim_idx], output_grad[idx]);
    }
}

// CUDA kernel for AdamW parameter update
__global__ void adamw_update_kernel(float* weights, float* gradients, float* m, float* v,
                                   float beta1, float beta2, float epsilon, 
                                   float learning_rate, float weight_decay,
                                   float alpha_t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float grad = gradients[idx];
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Update parameters with weight decay
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weights[idx] = weights[idx] * (1.0f - learning_rate * weight_decay) - update;
        
        // Zero gradient for next iteration
        gradients[idx] = 0.0f;
    }
}

// Initialize the language model
SLM* init_slm(int embedding_dim, int context_length, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store hyperparameters
    slm->embedding_dim = embedding_dim;
    slm->context_length = context_length;
    slm->batch_size = batch_size;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.001f;
    
    // Initialize cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    
    // Initialize SSM (state_dim = embedding_dim for simplicity)
    slm->ssm = init_ssm(embedding_dim, embedding_dim, embedding_dim, 1, batch_size);
    
    // Allocate host memory for data loading
    slm->h_chars = (int*)malloc(context_length * batch_size * sizeof(int));
    slm->h_next_chars = (int*)malloc(context_length * batch_size * sizeof(int));
    
    // Allocate device memory for characters
    CHECK_CUDA(cudaMalloc(&slm->d_chars, context_length * batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&slm->d_next_chars, context_length * batch_size * sizeof(int)));
    
    // Allocate device memory for model parameters
    CHECK_CUDA(cudaMalloc(&slm->d_embed, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj, embedding_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&slm->d_embed_grad, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_grad, embedding_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for Adam states
    CHECK_CUDA(cudaMalloc(&slm->d_embed_m, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embed_v, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_m, embedding_dim * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_v, embedding_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded, batch_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logits, batch_size * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logit_grad, batch_size * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_loss, sizeof(float)));
    
    // Initialize parameters on host
    float* h_embed = (float*)malloc(VOCAB_SIZE * embedding_dim * sizeof(float));
    float* h_proj = (float*)malloc(embedding_dim * VOCAB_SIZE * sizeof(float));
    
    // Xavier initialization
    float embed_scale = sqrtf(2.0f / (VOCAB_SIZE + embedding_dim));
    float proj_scale = sqrtf(2.0f / (embedding_dim + VOCAB_SIZE));
    
    for (int i = 0; i < VOCAB_SIZE * embedding_dim; i++) {
        h_embed[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * embed_scale;
    }
    
    for (int i = 0; i < embedding_dim * VOCAB_SIZE; i++) {
        h_proj[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * proj_scale;
    }
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(slm->d_embed, h_embed, VOCAB_SIZE * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_proj, h_proj, embedding_dim * VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam states to zero
    CHECK_CUDA(cudaMemset(slm->d_embed_m, 0, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embed_v, 0, VOCAB_SIZE * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_m, 0, embedding_dim * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_v, 0, embedding_dim * VOCAB_SIZE * sizeof(float)));
    
    free(h_embed);
    free(h_proj);
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
    if (!slm) return;
    
    free_ssm(slm->ssm);
    
    free(slm->h_chars);
    free(slm->h_next_chars);
    
    cudaFree(slm->d_chars);
    cudaFree(slm->d_next_chars);
    cudaFree(slm->d_embed);
    cudaFree(slm->d_proj);
    cudaFree(slm->d_embed_grad);
    cudaFree(slm->d_proj_grad);
    cudaFree(slm->d_embed_m);
    cudaFree(slm->d_embed_v);
    cudaFree(slm->d_proj_m);
    cudaFree(slm->d_proj_v);
    cudaFree(slm->d_embedded);
    cudaFree(slm->d_logits);
    cudaFree(slm->d_logit_grad);
    cudaFree(slm->d_loss);
    
    cublasDestroy(slm->cublas_handle);
    free(slm);
}

// Reset SSM states for new sequences
void reset_ssm_states(SLM* slm) {
    CHECK_CUDA(cudaMemset(slm->ssm->d_states, 0, slm->batch_size * slm->embedding_dim * sizeof(float)));
}

// Forward pass for single timestep
void forward_step(SLM* slm, int timestep) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Get characters for this timestep
    int* chars_t = slm->d_chars + timestep * slm->batch_size;
    
    // Embed characters
    int block_size = 256;
    int num_blocks = (slm->batch_size * slm->embedding_dim + block_size - 1) / block_size;
    embed_chars_kernel<<<num_blocks, block_size>>>(
        slm->d_embedded, slm->d_embed, chars_t, slm->batch_size, slm->embedding_dim
    );
    
    // Forward through SSM (single timestep)
    forward_pass_ssm(slm->ssm, slm->d_embedded);
    
    // Project to logits: logits = ssm_output * proj^T
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            VOCAB_SIZE, slm->batch_size, slm->embedding_dim,
                            &alpha, slm->d_proj, slm->embedding_dim,
                            slm->ssm->d_predictions, slm->embedding_dim,
                            &beta, slm->d_logits, VOCAB_SIZE));
}

// Compute loss for single timestep
float compute_loss(SLM* slm, int timestep) {
    // Get target characters for this timestep
    int* targets_t = slm->d_next_chars + timestep * slm->batch_size;
    
    // Reset loss buffer
    CHECK_CUDA(cudaMemset(slm->d_loss, 0, sizeof(float)));
    
    // Compute cross-entropy loss and gradients
    int block_size = 256;
    int num_blocks = (slm->batch_size + block_size - 1) / block_size;
    cross_entropy_kernel<<<num_blocks, block_size>>>(
        slm->d_loss, slm->d_logit_grad, slm->d_logits, targets_t, slm->batch_size
    );
    
    // Copy loss back to host
    float loss_val;
    CHECK_CUDA(cudaMemcpy(&loss_val, slm->d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    return loss_val;
}

// Backward pass for single timestep
void backward_step(SLM* slm, int timestep) {
    const float alpha = 1.0f;
    const float beta = 1.0f; // Accumulate gradients
    
    // Get characters for this timestep
    int* chars_t = slm->d_chars + timestep * slm->batch_size;
    
    // Gradient w.r.t. projection weights: d_proj_grad += ssm_output^T * d_logit_grad
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            slm->embedding_dim, VOCAB_SIZE, slm->batch_size,
                            &alpha, slm->ssm->d_predictions, slm->embedding_dim,
                            slm->d_logit_grad, VOCAB_SIZE,
                            &beta, slm->d_proj_grad, slm->embedding_dim));
    
    // Gradient w.r.t. SSM output: d_ssm_output = proj * d_logit_grad
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embedding_dim, slm->batch_size, VOCAB_SIZE,
                            &alpha, slm->d_proj, slm->embedding_dim,
                            slm->d_logit_grad, VOCAB_SIZE,
                            &alpha, slm->ssm->d_error, slm->embedding_dim));
    
    // Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_embedded);
    
    // Accumulate embedding gradients
    int block_size = 256;
    int num_blocks = (slm->batch_size * slm->embedding_dim + block_size - 1) / block_size;
    accumulate_embed_grad_kernel<<<num_blocks, block_size>>>(
        slm->d_embed_grad, slm->ssm->d_error, chars_t, slm->batch_size, slm->embedding_dim
    );
}

// Update weights using AdamW
void update_weights(SLM* slm, float learning_rate) {
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update embeddings
    int embed_size = VOCAB_SIZE * slm->embedding_dim;
    int embed_blocks = (embed_size + block_size - 1) / block_size;
    adamw_update_kernel<<<embed_blocks, block_size>>>(
        slm->d_embed, slm->d_embed_grad, slm->d_embed_m, slm->d_embed_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, embed_size
    );
    
    // Update projection weights
    int proj_size = slm->embedding_dim * VOCAB_SIZE;
    int proj_blocks = (proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<proj_blocks, block_size>>>(
        slm->d_proj, slm->d_proj_grad, slm->d_proj_m, slm->d_proj_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, proj_size
    );
    
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
    
    // Zero SSM gradients
    zero_gradients_ssm(slm->ssm);
}

// Train for one batch
float train_batch(SLM* slm, TextData* text_data, float learning_rate) {
    // Get batch data
    get_batch(text_data, slm->h_chars, slm->h_next_chars, slm->batch_size, slm->context_length);
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(slm->d_chars, slm->h_chars, 
                         slm->context_length * slm->batch_size * sizeof(int), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_next_chars, slm->h_next_chars, 
                         slm->context_length * slm->batch_size * sizeof(int), 
                         cudaMemcpyHostToDevice));
    
    // Reset states for new sequences
    reset_ssm_states(slm);
    
    // Zero gradients
    CHECK_CUDA(cudaMemset(slm->d_embed_grad, 0, VOCAB_SIZE * slm->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_grad, 0, slm->embedding_dim * VOCAB_SIZE * sizeof(float)));
    zero_gradients_ssm(slm->ssm);
    
    float total_loss = 0.0f;
    
    // Process each timestep
    for (int t = 0; t < slm->context_length; t++) {
        // Forward pass
        forward_step(slm, t);
        
        // Compute loss
        float step_loss = compute_loss(slm, t);
        total_loss += step_loss;
        
        // Backward pass
        backward_step(slm, t);
    }
    
    // Update weights
    update_weights(slm, learning_rate);
    
    return total_loss / slm->context_length;
}

// Generate text (simple greedy decoding)
void generate_text(SLM* slm, const char* prompt, int length) {
    printf("Generating text with prompt: '%s'\n", prompt);
    
    // Reset states
    reset_ssm_states(slm);
    
    // Process prompt
    int prompt_len = strlen(prompt);
    for (int i = 0; i < prompt_len; i++) {
        int char_val = (int)(unsigned char)prompt[i];
        
        // Create single-character batch
        int h_char = char_val;
        CHECK_CUDA(cudaMemcpy(slm->d_chars, &h_char, sizeof(int), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_step(slm, 0);
    }
    
    // Generate new characters
    printf("Generated: ");
    for (int i = 0; i < length; i++) {
        // Get logits from last forward pass
        float* h_logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_logits, slm->d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Simple greedy sampling (pick max)
        int best_char = 0;
        float best_score = h_logits[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (h_logits[j] > best_score) {
                best_score = h_logits[j];
                best_char = j;
            }
        }
        
        printf("%c", (char)best_char);
        fflush(stdout);
        
        // Feed back as input
        CHECK_CUDA(cudaMemcpy(slm->d_chars, &best_char, sizeof(int), cudaMemcpyHostToDevice));
        forward_step(slm, 0);
        
        free(h_logits);
    }
    printf("\n");
}

#endif