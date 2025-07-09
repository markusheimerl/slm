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
    // Embedded SSM instance
    SSM* ssm;
    
    // Device pointers for language model matrices
    float* d_embeddings;        // vocab_size x embedding_dim
    float* d_output_weights;    // embedding_dim x vocab_size
    
    // Device pointers for gradients
    float* d_embed_grad;        // vocab_size x embedding_dim
    float* d_output_grad;       // embedding_dim x vocab_size
    
    // Device pointers for Adam parameters
    float* d_embed_m; float* d_embed_v;
    float* d_output_m; float* d_output_v;
    
    // Adam hyperparameters
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays
    float* d_embedded;          // seq_len x batch_size x embedding_dim
    float* d_logits;           // seq_len x batch_size x vocab_size
    float* d_loss_buffer;      // single loss value
    float* d_logit_grad;       // seq_len x batch_size x vocab_size
    float* d_softmax_buffer;   // seq_len x batch_size x vocab_size
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
    int seq_len;
    int batch_size;
} SLM;

// CUDA kernel for embedding lookup
__global__ void embedding_lookup_kernel(float* output, const float* embeddings, 
                                       const int* indices, int seq_len, int batch_size, 
                                       int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        if (token_id >= 0 && token_id < vocab_size) {
            output[idx] = embeddings[token_id * embedding_dim + dim];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// CUDA kernel for embedding gradient accumulation
__global__ void embedding_grad_kernel(float* embed_grad, const float* output_grad, 
                                     const int* indices, int seq_len, int batch_size, 
                                     int embedding_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        if (token_id >= 0 && token_id < vocab_size) {
            atomicAdd(&embed_grad[token_id * embedding_dim + dim], output_grad[idx]);
        }
    }
}

// CUDA kernel for cross-entropy loss and gradient computation
__global__ void cross_entropy_kernel(float* loss, float* grad, const float* logits, 
                                    const int* targets, int seq_len, int batch_size, 
                                    int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = seq_len * batch_size;
    
    if (idx < total_positions) {
        int seq_pos = idx / batch_size;
        int batch_pos = idx % batch_size;
        int logit_offset = seq_pos * batch_size * vocab_size + batch_pos * vocab_size;
        int target = targets[idx];
        
        if (target >= 0 && target < vocab_size) {
            const float* logit_ptr = logits + logit_offset;
            float* grad_ptr = grad + logit_offset;
            
            // Find max for numerical stability
            float max_logit = logit_ptr[0];
            for (int i = 1; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logit_ptr[i]);
            }
            
            // Compute softmax and loss
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logit_ptr[i] - max_logit);
            }
            
            float log_sum = logf(sum_exp);
            float loss_val = -logit_ptr[target] + max_logit + log_sum;
            atomicAdd(loss, loss_val);
            
            // Compute gradient (softmax - one_hot)
            for (int i = 0; i < vocab_size; i++) {
                float prob = expf(logit_ptr[i] - max_logit) / sum_exp;
                grad_ptr[i] = prob - (i == target ? 1.0f : 0.0f);
            }
        }
    }
}

// CUDA kernel for AdamW parameter update
__global__ void adamw_update_kernel(float* weights, float* gradients, float* m, float* v,
                                   float beta1, float beta2, float epsilon, 
                                   float learning_rate, float weight_decay,
                                   float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx] / batch_size;
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Update parameters with weight decay
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weights[idx] = weights[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Initialize the language model
SLM* init_slm(int vocab_size, int embedding_dim, int seq_len, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->vocab_size = vocab_size;
    slm->embedding_dim = embedding_dim;
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;
    
    // Initialize embedded SSM
    slm->ssm = init_ssm(embedding_dim, 256, embedding_dim, seq_len, batch_size);
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.001f;
    
    // Initialize cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    
    // Allocate host memory for initialization
    float* h_embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* h_output_weights = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    // Xavier initialization
    float embed_scale = sqrtf(2.0f / (vocab_size + embedding_dim));
    float output_scale = sqrtf(2.0f / (embedding_dim + vocab_size));
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        h_embeddings[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * embed_scale;
    }
    
    for (int i = 0; i < embedding_dim * vocab_size; i++) {
        h_output_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * output_scale;
    }
    
    // Allocate device memory for matrices
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_weights, embedding_dim * vocab_size * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&slm->d_embed_grad, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_grad, embedding_dim * vocab_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&slm->d_embed_m, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embed_v, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_m, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_v, embedding_dim * vocab_size * sizeof(float)));
    
    // Allocate device memory for helper arrays
    CHECK_CUDA(cudaMalloc(&slm->d_embedded, seq_len * batch_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logits, seq_len * batch_size * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_loss_buffer, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logit_grad, seq_len * batch_size * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax_buffer, seq_len * batch_size * vocab_size * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                         vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_weights, h_output_weights, 
                         embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_embed_m, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embed_v, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_m, 0, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_v, 0, embedding_dim * vocab_size * sizeof(float)));
    
    free(h_embeddings);
    free(h_output_weights);
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
    if (!slm) return;
    
    // Free embedded SSM
    free_ssm(slm->ssm);
    
    // Free device memory
    cudaFree(slm->d_embeddings);
    cudaFree(slm->d_output_weights);
    cudaFree(slm->d_embed_grad);
    cudaFree(slm->d_output_grad);
    cudaFree(slm->d_embed_m);
    cudaFree(slm->d_embed_v);
    cudaFree(slm->d_output_m);
    cudaFree(slm->d_output_v);
    cudaFree(slm->d_embedded);
    cudaFree(slm->d_logits);
    cudaFree(slm->d_loss_buffer);
    cudaFree(slm->d_logit_grad);
    cudaFree(slm->d_softmax_buffer);
    
    // Destroy cuBLAS handle
    cublasDestroy(slm->cublas_handle);
    
    free(slm);
}

// Forward pass
void forward_pass_slm(SLM* slm, int* d_input_ids) {
    // Embedding lookup
    int block_size = 256;
    int total_elements = slm->seq_len * slm->batch_size * slm->embedding_dim;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    embedding_lookup_kernel<<<num_blocks, block_size>>>(
        slm->d_embedded, slm->d_embeddings, d_input_ids,
        slm->seq_len, slm->batch_size, slm->embedding_dim, slm->vocab_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Run through SSM
    forward_pass_ssm(slm->ssm, slm->d_embedded);
}

// Calculate loss
float calculate_loss_slm(SLM* slm, int* d_target_ids) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Output projection: logits = ssm_output * output_weights^T
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            slm->vocab_size, slm->seq_len * slm->batch_size, slm->embedding_dim,
                            &alpha, slm->d_output_weights, slm->embedding_dim,
                            slm->ssm->d_predictions, slm->embedding_dim,
                            &beta, slm->d_logits, slm->vocab_size));
    
    // Cross-entropy loss computation
    CHECK_CUDA(cudaMemset(slm->d_loss_buffer, 0, sizeof(float)));
    
    int block_size = 256;
    int total_positions = slm->seq_len * slm->batch_size;
    int num_blocks = (total_positions + block_size - 1) / block_size;
    
    cross_entropy_kernel<<<num_blocks, block_size>>>(
        slm->d_loss_buffer, slm->d_logit_grad, slm->d_logits, d_target_ids,
        slm->seq_len, slm->batch_size, slm->vocab_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy loss back to host
    float loss_val;
    CHECK_CUDA(cudaMemcpy(&loss_val, slm->d_loss_buffer, sizeof(float), cudaMemcpyDeviceToHost));
    
    return loss_val / total_positions;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    CHECK_CUDA(cudaMemset(slm->d_embed_grad, 0, slm->vocab_size * slm->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_grad, 0, slm->embedding_dim * slm->vocab_size * sizeof(float)));
    zero_gradients_ssm(slm->ssm);
}

// Backward pass
void backward_pass_slm(SLM* slm, int* d_input_ids) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;
    
    // Gradient w.r.t. output weights: grad = ssm_output^T * logit_grad
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            slm->embedding_dim, slm->vocab_size, slm->seq_len * slm->batch_size,
                            &alpha, slm->ssm->d_predictions, slm->embedding_dim,
                            slm->d_logit_grad, slm->vocab_size,
                            &beta_add, slm->d_output_grad, slm->embedding_dim));
    
    // Gradient w.r.t. SSM output: grad = output_weights * logit_grad
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embedding_dim, slm->seq_len * slm->batch_size, slm->vocab_size,
                            &alpha, slm->d_output_weights, slm->embedding_dim,
                            slm->d_logit_grad, slm->vocab_size,
                            &beta, slm->ssm->d_error, slm->embedding_dim));
    
    // Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_embedded);
    
    // Gradient w.r.t. embeddings
    int block_size = 256;
    int total_elements = slm->seq_len * slm->batch_size * slm->embedding_dim;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    embedding_grad_kernel<<<num_blocks, block_size>>>(
        slm->d_embed_grad, slm->ssm->d_error, d_input_ids,
        slm->seq_len, slm->batch_size, slm->embedding_dim, slm->vocab_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
}

// Update weights using AdamW
void update_weights_slm(SLM* slm, float learning_rate) {
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update embeddings
    int embed_size = slm->vocab_size * slm->embedding_dim;
    int embed_blocks = (embed_size + block_size - 1) / block_size;
    
    adamw_update_kernel<<<embed_blocks, block_size>>>(
        slm->d_embeddings, slm->d_embed_grad, slm->d_embed_m, slm->d_embed_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, embed_size, slm->batch_size
    );
    
    // Update output weights
    int output_size = slm->embedding_dim * slm->vocab_size;
    int output_blocks = (output_size + block_size - 1) / block_size;
    
    adamw_update_kernel<<<output_blocks, block_size>>>(
        slm->d_output_weights, slm->d_output_grad, slm->d_output_m, slm->d_output_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, output_size, slm->batch_size
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Allocate temporary host memory
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* h_output_weights = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, 
                         slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_weights, slm->d_output_weights, 
                         slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        free(h_embeddings);
        free(h_output_weights);
        return;
    }
    
    // Save dimensions
    fwrite(&slm->vocab_size, sizeof(int), 1, file);
    fwrite(&slm->embedding_dim, sizeof(int), 1, file);
    fwrite(&slm->seq_len, sizeof(int), 1, file);
    fwrite(&slm->batch_size, sizeof(int), 1, file);
    fwrite(&slm->t, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(h_output_weights, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    
    // Save Adam state
    float* h_embed_m = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* h_embed_v = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* h_output_m = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    float* h_output_v = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_embed_m, slm->d_embed_m, 
                         slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embed_v, slm->d_embed_v, 
                         slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_m, slm->d_output_m, 
                         slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_v, slm->d_output_v, 
                         slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_embed_m, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(h_embed_v, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(h_output_m, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    fwrite(h_output_v, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    
    free(h_embeddings);
    free(h_output_weights);
    free(h_embed_m);
    free(h_embed_v);
    free(h_output_m);
    free(h_output_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int vocab_size, embedding_dim, seq_len, stored_batch_size, t;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SLM* slm = init_slm(vocab_size, embedding_dim, seq_len, batch_size);
    slm->t = t;
    
    // Allocate temporary host memory
    float* h_embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* h_output_weights = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    // Load matrices
    fread(h_embeddings, sizeof(float), vocab_size * embedding_dim, file);
    fread(h_output_weights, sizeof(float), embedding_dim * vocab_size, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                         vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_weights, h_output_weights, 
                         embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* h_embed_m = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* h_embed_v = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* h_output_m = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    float* h_output_v = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    fread(h_embed_m, sizeof(float), vocab_size * embedding_dim, file);
    fread(h_embed_v, sizeof(float), vocab_size * embedding_dim, file);
    fread(h_output_m, sizeof(float), embedding_dim * vocab_size, file);
    fread(h_output_v, sizeof(float), embedding_dim * vocab_size, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(slm->d_embed_m, h_embed_m, 
                         vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embed_v, h_embed_v, 
                         vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_m, h_output_m, 
                         embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_v, h_output_v, 
                         embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_embeddings);
    free(h_output_weights);
    free(h_embed_m);
    free(h_embed_v);
    free(h_output_m);
    free(h_output_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return slm;
}

#endif