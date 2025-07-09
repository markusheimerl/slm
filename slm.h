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
    float* d_embed;      // VOCAB_SIZE × embed_dim
    float* d_proj;       // embed_dim × VOCAB_SIZE
    
    // Gradients (on device)
    float* d_embed_grad; // VOCAB_SIZE × embed_dim
    float* d_proj_grad;  // embed_dim × VOCAB_SIZE
    
    // Adam states for embedding and projection
    float* d_embed_m, *d_embed_v;
    float* d_proj_m, *d_proj_v;
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Working buffers for single timestep
    float* d_X;          // batch_size × embed_dim (input embeddings)
    float* d_logits;     // batch_size × VOCAB_SIZE (output logits)
    float* d_probs;      // batch_size × VOCAB_SIZE (softmax probabilities)
    float* d_targets;    // batch_size × VOCAB_SIZE (one-hot targets)
    float* d_loss;       // scalar loss value
    
    // Character buffers
    int* h_chars;        // batch_size current characters
    int* h_next_chars;   // batch_size next characters
    int* d_chars;        // batch_size current characters (device)
    int* d_next_chars;   // batch_size next characters (device)
    
    // Corpus positions for each batch element
    size_t* positions;   // batch_size positions in corpus
    
    // Hyperparameters
    int embed_dim;
    int batch_size;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} SLM;

// CUDA kernels
__global__ void embed_chars_kernel(float* X, const float* embed_matrix, 
                                  const int* chars, int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * embed_dim) {
        int batch_idx = idx / embed_dim;
        int dim_idx = idx % embed_dim;
        int char_val = chars[batch_idx];
        X[idx] = embed_matrix[char_val * embed_dim + dim_idx];
    }
}

__global__ void create_onehot_kernel(float* targets, const int* chars, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * VOCAB_SIZE) {
        int batch_idx = idx / VOCAB_SIZE;
        int vocab_idx = idx % VOCAB_SIZE;
        int target_char = chars[batch_idx];
        targets[idx] = (vocab_idx == target_char) ? 1.0f : 0.0f;
    }
}

__global__ void softmax_kernel(float* probs, const float* logits, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        const float* logit_row = logits + batch_idx * VOCAB_SIZE;
        float* prob_row = probs + batch_idx * VOCAB_SIZE;
        
        // Find max for numerical stability
        float max_logit = logit_row[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            max_logit = fmaxf(max_logit, logit_row[i]);
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            prob_row[i] = expf(logit_row[i] - max_logit);
            sum_exp += prob_row[i];
        }
        
        // Normalize
        for (int i = 0; i < VOCAB_SIZE; i++) {
            prob_row[i] /= sum_exp;
        }
    }
}

__global__ void cross_entropy_loss_kernel(float* loss, float* grad_logits, 
                                         const float* probs, const float* targets, 
                                         int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * VOCAB_SIZE) {
        int batch_idx = idx / VOCAB_SIZE;
        int vocab_idx = idx % VOCAB_SIZE;
        
        float prob = probs[idx];
        float target = targets[idx];
        
        // Gradient: prob - target
        grad_logits[idx] = prob - target;
        
        // Loss contribution (only for target class)
        if (target > 0.5f) {
            atomicAdd(loss, -logf(fmaxf(prob, 1e-8f)));
        }
    }
}

__global__ void accumulate_embed_grad_kernel(float* embed_grad, const float* grad_X,
                                           const int* chars, int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * embed_dim) {
        int batch_idx = idx / embed_dim;
        int dim_idx = idx % embed_dim;
        int char_val = chars[batch_idx];
        atomicAdd(&embed_grad[char_val * embed_dim + dim_idx], grad_X[idx]);
    }
}

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

// Function declarations
SLM* init_slm(int embed_dim, int batch_size, TextData* text_data);
void free_slm(SLM* slm);
void extract_chars(SLM* slm, TextData* text_data);
void advance_positions(SLM* slm, TextData* text_data);
void forward_pass(SLM* slm);
float compute_loss(SLM* slm);
void backward_pass(SLM* slm);
void update_weights(SLM* slm, float learning_rate);
float train_step(SLM* slm, TextData* text_data, float learning_rate);
void generate_text(SLM* slm, const char* prompt, int length);

// Initialize the language model
SLM* init_slm(int embed_dim, int batch_size, TextData* text_data) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store hyperparameters
    slm->embed_dim = embed_dim;
    slm->batch_size = batch_size;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;
    
    // Initialize cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    
    // Initialize SSM with seq_len=1 for single timestep processing
    slm->ssm = init_ssm(embed_dim, embed_dim, embed_dim, 1, batch_size);
    
    // Initialize random positions in corpus
    slm->positions = (size_t*)malloc(batch_size * sizeof(size_t));
    for (int i = 0; i < batch_size; i++) {
        slm->positions[i] = rand() % (text_data->text_size - 1);
    }
    
    // Allocate host memory
    slm->h_chars = (int*)malloc(batch_size * sizeof(int));
    slm->h_next_chars = (int*)malloc(batch_size * sizeof(int));
    
    // Allocate device memory for characters
    CHECK_CUDA(cudaMalloc(&slm->d_chars, batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&slm->d_next_chars, batch_size * sizeof(int)));
    
    // Allocate device memory for model parameters
    CHECK_CUDA(cudaMalloc(&slm->d_embed, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj, embed_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&slm->d_embed_grad, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_grad, embed_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for Adam states
    CHECK_CUDA(cudaMalloc(&slm->d_embed_m, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embed_v, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_m, embed_dim * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_proj_v, embed_dim * VOCAB_SIZE * sizeof(float)));
    
    // Allocate device memory for working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_X, batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logits, batch_size * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_probs, batch_size * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_targets, batch_size * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_loss, sizeof(float)));
    
    // Initialize parameters on host
    float* h_embed = (float*)malloc(VOCAB_SIZE * embed_dim * sizeof(float));
    float* h_proj = (float*)malloc(embed_dim * VOCAB_SIZE * sizeof(float));
    
    // Xavier initialization with smaller scale
    float embed_scale = sqrtf(1.0f / embed_dim);
    float proj_scale = sqrtf(1.0f / embed_dim);
    
    for (int i = 0; i < VOCAB_SIZE * embed_dim; i++) {
        h_embed[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * embed_scale;
    }
    
    for (int i = 0; i < embed_dim * VOCAB_SIZE; i++) {
        h_proj[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * proj_scale;
    }
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(slm->d_embed, h_embed, VOCAB_SIZE * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_proj, h_proj, embed_dim * VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam states to zero
    CHECK_CUDA(cudaMemset(slm->d_embed_m, 0, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embed_v, 0, VOCAB_SIZE * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_m, 0, embed_dim * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_v, 0, embed_dim * VOCAB_SIZE * sizeof(float)));
    
    free(h_embed);
    free(h_proj);
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
    if (!slm) return;
    
    free_ssm(slm->ssm);
    
    free(slm->positions);
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
    cudaFree(slm->d_X);
    cudaFree(slm->d_logits);
    cudaFree(slm->d_probs);
    cudaFree(slm->d_targets);
    cudaFree(slm->d_loss);
    
    cublasDestroy(slm->cublas_handle);
    free(slm);
}

// Extract characters at current positions
void extract_chars(SLM* slm, TextData* text_data) {
    for (int i = 0; i < slm->batch_size; i++) {
        size_t pos = slm->positions[i];
        if (pos >= text_data->text_size - 1) {
            pos = 0; // Wrap around
            slm->positions[i] = pos;
        }
        
        slm->h_chars[i] = (int)(unsigned char)text_data->text[pos];
        slm->h_next_chars[i] = (int)(unsigned char)text_data->text[pos + 1];
    }
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(slm->d_chars, slm->h_chars, slm->batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_next_chars, slm->h_next_chars, slm->batch_size * sizeof(int), cudaMemcpyHostToDevice));
}

// Advance positions
void advance_positions(SLM* slm, TextData* text_data) {
    for (int i = 0; i < slm->batch_size; i++) {
        slm->positions[i] = (slm->positions[i] + 1) % (text_data->text_size - 1);
    }
}

// Forward pass
void forward_pass(SLM* slm) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Embed characters
    int block_size = 256;
    int embed_blocks = (slm->batch_size * slm->embed_dim + block_size - 1) / block_size;
    
    embed_chars_kernel<<<embed_blocks, block_size>>>(
        slm->d_X, slm->d_embed, slm->d_chars, slm->batch_size, slm->embed_dim
    );
    
    // Forward through SSM (single timestep)
    forward_pass_ssm(slm->ssm, slm->d_X);
    
    // Project to logits: logits = ssm_output * proj^T
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            VOCAB_SIZE, slm->batch_size, slm->embed_dim,
                            &alpha, slm->d_proj, slm->embed_dim,
                            slm->ssm->d_predictions, slm->embed_dim,
                            &beta, slm->d_logits, VOCAB_SIZE));
    
    // Compute softmax
    int softmax_blocks = (slm->batch_size + block_size - 1) / block_size;
    softmax_kernel<<<softmax_blocks, block_size>>>(
        slm->d_probs, slm->d_logits, slm->batch_size
    );
}

// Compute loss and gradients
float compute_loss(SLM* slm) {
    int block_size = 256;
    
    // Create one-hot targets
    int onehot_blocks = (slm->batch_size * VOCAB_SIZE + block_size - 1) / block_size;
    create_onehot_kernel<<<onehot_blocks, block_size>>>(
        slm->d_targets, slm->d_next_chars, slm->batch_size
    );
    
    // Reset loss
    CHECK_CUDA(cudaMemset(slm->d_loss, 0, sizeof(float)));
    
    // Compute loss and gradients
    cross_entropy_loss_kernel<<<onehot_blocks, block_size>>>(
        slm->d_loss, slm->d_logits, slm->d_probs, slm->d_targets, slm->batch_size
    );
    
    // Get loss value
    float loss_val;
    CHECK_CUDA(cudaMemcpy(&loss_val, slm->d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    return loss_val / slm->batch_size;
}

// Backward pass
void backward_pass(SLM* slm) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    
    // Gradient w.r.t. projection weights: d_proj_grad += ssm_output^T * d_logits
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            slm->embed_dim, VOCAB_SIZE, slm->batch_size,
                            &alpha, slm->ssm->d_predictions, slm->embed_dim,
                            slm->d_logits, VOCAB_SIZE,
                            &beta, slm->d_proj_grad, slm->embed_dim));
    
    // Gradient w.r.t. SSM output: d_error = proj * d_logits
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embed_dim, slm->batch_size, VOCAB_SIZE,
                            &alpha, slm->d_proj, slm->embed_dim,
                            slm->d_logits, VOCAB_SIZE,
                            &alpha, slm->ssm->d_error, slm->embed_dim));
    
    // Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_X);
    
    // Accumulate embedding gradients
    int block_size = 256;
    int embed_blocks = (slm->batch_size * slm->embed_dim + block_size - 1) / block_size;
    
    accumulate_embed_grad_kernel<<<embed_blocks, block_size>>>(
        slm->d_embed_grad, slm->ssm->d_error, slm->d_chars, slm->batch_size, slm->embed_dim
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
    int embed_size = VOCAB_SIZE * slm->embed_dim;
    int embed_blocks = (embed_size + block_size - 1) / block_size;
    adamw_update_kernel<<<embed_blocks, block_size>>>(
        slm->d_embed, slm->d_embed_grad, slm->d_embed_m, slm->d_embed_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, embed_size
    );
    
    // Update projection weights
    int proj_size = slm->embed_dim * VOCAB_SIZE;
    int proj_blocks = (proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<proj_blocks, block_size>>>(
        slm->d_proj, slm->d_proj_grad, slm->d_proj_m, slm->d_proj_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, proj_size
    );
    
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
}

// Train one step
float train_step(SLM* slm, TextData* text_data, float learning_rate) {
    // Extract characters at current positions
    extract_chars(slm, text_data);
    
    // Zero gradients
    CHECK_CUDA(cudaMemset(slm->d_embed_grad, 0, VOCAB_SIZE * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_proj_grad, 0, slm->embed_dim * VOCAB_SIZE * sizeof(float)));
    zero_gradients_ssm(slm->ssm);
    
    // Forward pass
    forward_pass(slm);
    
    // Compute loss
    float loss = compute_loss(slm);
    
    // Backward pass
    backward_pass(slm);
    
    // Update weights
    update_weights(slm, learning_rate);
    
    // Advance positions
    advance_positions(slm, text_data);
    
    return loss;
}

// Generate text
void generate_text(SLM* slm, const char* prompt, int length) {
    printf("Generating text with prompt: '%s'\n", prompt);
    
    // Reset SSM state
    CHECK_CUDA(cudaMemset(slm->ssm->d_states, 0, slm->batch_size * slm->embed_dim * sizeof(float)));
    
    // Process prompt (use first batch slot)
    int prompt_len = strlen(prompt);
    for (int i = 0; i < prompt_len; i++) {
        int char_val = (int)(unsigned char)prompt[i];
        
        // Set character for first batch slot
        CHECK_CUDA(cudaMemcpy(slm->d_chars, &char_val, sizeof(int), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass(slm);
    }
    
    // Generate new characters
    printf("Generated: ");
    for (int i = 0; i < length; i++) {
        // Get probabilities from logits
        float* h_probs = (float*)malloc(VOCAB_SIZE * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_probs, slm->d_probs, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Simple greedy sampling (pick max)
        int best_char = 0;
        float best_prob = h_probs[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (h_probs[j] > best_prob) {
                best_prob = h_probs[j];
                best_char = j;
            }
        }
        
        printf("%c", (char)best_char);
        fflush(stdout);
        
        // Feed back as input
        CHECK_CUDA(cudaMemcpy(slm->d_chars, &best_char, sizeof(int), cudaMemcpyHostToDevice));
        forward_pass(slm);
        
        free(h_probs);
    }
    printf("\n");
}

#endif