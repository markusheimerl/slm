#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ssm/gpu/ssm.h"

// CUDA Error checking macro (reuse from SSM)
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

// cuBLAS Error checking macro (reuse from SSM)
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
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays
    float* d_embedded;          // seq_len x batch_size x embedding_dim
    float* d_logits;           // seq_len x batch_size x vocab_size
    float* d_loss;             // single loss value
    float* d_logit_grad;       // seq_len x batch_size x vocab_size
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
    int seq_len;
    int batch_size;
} SLM;

// Initialize the language model
SLM* init_slm(int vocab_size, int embedding_dim, int seq_len, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->vocab_size = vocab_size;
    slm->embedding_dim = embedding_dim;
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;
    
    // Initialize embedded SSM (embedding_dim as both input and output)
    slm->ssm = init_ssm(embedding_dim, 256, embedding_dim, seq_len, batch_size);
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.001f;
    
    // Allocate host memory for initialization
    float* embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* output_weights = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    // Xavier initialization
    float embed_scale = sqrtf(2.0f / (vocab_size + embedding_dim));
    float output_scale = sqrtf(2.0f / (embedding_dim + vocab_size));
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        embeddings[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * embed_scale;
    }
    
    for (int i = 0; i < embedding_dim * vocab_size; i++) {
        output_weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * output_scale;
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
    CHECK_CUDA(cudaMalloc(&slm->d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logit_grad, seq_len * batch_size * vocab_size * sizeof(float)));
    
    // Copy initialized matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, embeddings, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_weights, output_weights, embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_embed_m, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embed_v, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_m, 0, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_v, 0, embedding_dim * vocab_size * sizeof(float)));
    
    // Free host memory
    free(embeddings);
    free(output_weights);
    
    return slm;
}

// Free memory
void free_slm(SLM* slm) {
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
    cudaFree(slm->d_loss); 
    cudaFree(slm->d_logit_grad);
    
    free(slm);
}

// CUDA kernel for embedding lookup
__global__ void embedding_forward_kernel_slm(float* output, float* embeddings, int* indices,
                                             int seq_len, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        if (token_id >= 0 && token_id < 256) {  // Bounds check
            output[idx] = embeddings[token_id * embedding_dim + dim];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// CUDA kernel for embedding backward
__global__ void embedding_backward_kernel_slm(float* embed_grad, float* output_grad, int* indices,
                                              int seq_len, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        if (token_id >= 0 && token_id < 256) {  // Bounds check
            atomicAdd(&embed_grad[token_id * embedding_dim + dim], output_grad[idx]);
        }
    }
}

// CUDA kernel for cross-entropy loss forward
__global__ void cross_entropy_forward_kernel_slm(float* loss, float* grad, float* logits, int* targets,
                                                 int seq_len, int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = seq_len * batch_size;
    
    if (idx < total_positions) {
        float* logit_ptr = logits + idx * vocab_size;
        int target = targets[idx];
        
        if (target >= 0 && target < vocab_size) {  // Bounds check
            // Compute max for numerical stability
            float max_logit = logit_ptr[0];
            for (int i = 1; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logit_ptr[i]);
            }
            
            // Compute softmax denominator
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logit_ptr[i] - max_logit);
            }
            
            // Compute loss for this position
            float position_loss = -logit_ptr[target] + max_logit + logf(sum_exp);
            atomicAdd(loss, position_loss);
            
            // Compute gradient
            float* grad_ptr = grad + idx * vocab_size;
            for (int i = 0; i < vocab_size; i++) {
                float prob = expf(logit_ptr[i] - max_logit) / sum_exp;
                grad_ptr[i] = prob;
                if (i == target) {
                    grad_ptr[i] -= 1.0f;
                }
            }
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

// Forward pass
void forward_pass_slm(SLM* slm, int* d_input_ids) {
    // Embedding lookup
    int block_size = 256;
    int num_blocks = (slm->seq_len * slm->batch_size * slm->embedding_dim + block_size - 1) / block_size;
    embedding_forward_kernel_slm<<<num_blocks, block_size>>>(
        slm->d_embedded, slm->d_embeddings, d_input_ids,
        slm->seq_len, slm->batch_size, slm->embedding_dim
    );
    
    // Run through SSM
    forward_pass_ssm(slm->ssm, slm->d_embedded);
}

// Calculate loss
float calculate_loss_slm(SLM* slm, int* d_target_ids) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Output projection: logits = predictions * output_weights^T
    CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            slm->vocab_size, slm->seq_len * slm->batch_size, slm->embedding_dim,
                            &alpha, slm->d_output_weights, slm->embedding_dim,
                            slm->ssm->d_predictions, slm->embedding_dim,
                            &beta, slm->d_logits, slm->vocab_size));
    
    // Cross-entropy loss
    CHECK_CUDA(cudaMemset(slm->d_loss, 0, sizeof(float)));
    int block_size = 256;
    int num_blocks = (slm->seq_len * slm->batch_size + block_size - 1) / block_size;
    cross_entropy_forward_kernel_slm<<<num_blocks, block_size>>>(
        slm->d_loss, slm->d_logit_grad, slm->d_logits, d_target_ids,
        slm->seq_len, slm->batch_size, slm->vocab_size
    );
    
    float loss_val;
    CHECK_CUDA(cudaMemcpy(&loss_val, slm->d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return loss_val / (slm->seq_len * slm->batch_size);
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
    
    // Output weight gradient
    CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            slm->embedding_dim, slm->vocab_size, slm->seq_len * slm->batch_size,
                            &alpha, slm->ssm->d_predictions, slm->embedding_dim,
                            slm->d_logit_grad, slm->vocab_size,
                            &beta_add, slm->d_output_grad, slm->embedding_dim));
    
    // SSM output gradient
    CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embedding_dim, slm->seq_len * slm->batch_size, slm->vocab_size,
                            &alpha, slm->d_output_weights, slm->embedding_dim,
                            slm->d_logit_grad, slm->vocab_size,
                            &beta, slm->ssm->d_error, slm->embedding_dim));
    
    // Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_embedded);
    
    // Embedding gradient
    int block_size = 256;
    int num_blocks = (slm->seq_len * slm->batch_size * slm->embedding_dim + block_size - 1) / block_size;
    embedding_backward_kernel_slm<<<num_blocks, block_size>>>(
        slm->d_embed_grad, slm->ssm->d_error, d_input_ids,
        slm->seq_len, slm->batch_size, slm->embedding_dim
    );
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
    adamw_update_kernel_slm<<<embed_blocks, block_size>>>(
        slm->d_embeddings, slm->d_embed_grad, slm->d_embed_m, slm->d_embed_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, embed_size, slm->batch_size
    );
    
    // Update output weights
    int output_size = slm->embedding_dim * slm->vocab_size;
    int output_blocks = (output_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<output_blocks, block_size>>>(
        slm->d_output_weights, slm->d_output_grad, slm->d_output_m, slm->d_output_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, output_size, slm->batch_size
    );
    
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
}

// Load text and prepare sequences (streaming approach for memory efficiency)
void load_text_sequences_stream(const char* filename, int** d_input_ids, int** d_target_ids, 
                               int* num_batches, int seq_len, int batch_size, int max_batches) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        exit(1);
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Calculate number of sequences we can create
    int total_sequences = (file_size - 1) / seq_len;
    int total_batches = total_sequences / batch_size;
    
    // Limit to max_batches to save memory
    *num_batches = (max_batches > 0 && max_batches < total_batches) ? max_batches : total_batches;
    int used_sequences = (*num_batches) * batch_size;
    
    printf("File size: %ld bytes, limiting to %d batches of %d sequences\n", 
           file_size, *num_batches, batch_size);
    
    // Allocate memory for the limited number of sequences
    int* h_input_ids = (int*)malloc(used_sequences * seq_len * sizeof(int));
    int* h_target_ids = (int*)malloc(used_sequences * seq_len * sizeof(int));
    
    // Read and process sequences
    for (int seq = 0; seq < used_sequences; seq++) {
        unsigned char buffer[seq_len + 1];
        fread(buffer, 1, seq_len + 1, file);
        
        for (int pos = 0; pos < seq_len; pos++) {
            h_input_ids[seq * seq_len + pos] = buffer[pos];
            h_target_ids[seq * seq_len + pos] = buffer[pos + 1];
        }
    }
    
    fclose(file);
    
    // Allocate and copy to device
    CHECK_CUDA(cudaMalloc(d_input_ids, used_sequences * seq_len * sizeof(int)));
    CHECK_CUDA(cudaMalloc(d_target_ids, used_sequences * seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(*d_input_ids, h_input_ids, used_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_target_ids, h_target_ids, used_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    free(h_input_ids);
    free(h_target_ids);
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Allocate temporary host memory
    float* embeddings = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* output_weights = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    
    // Copy matrices from device to host
    CHECK_CUDA(cudaMemcpy(embeddings, slm->d_embeddings, slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(output_weights, slm->d_output_weights, slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(embeddings); free(output_weights);
        return;
    }
    
    // Save dimensions
    fwrite(&slm->vocab_size, sizeof(int), 1, file);
    fwrite(&slm->embedding_dim, sizeof(int), 1, file);
    fwrite(&slm->seq_len, sizeof(int), 1, file);
    fwrite(&slm->batch_size, sizeof(int), 1, file);
    
    // Save matrices
    fwrite(embeddings, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(output_weights, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    
    fwrite(&slm->t, sizeof(int), 1, file);
    
    // Save Adam state
    float* embed_m = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* embed_v = (float*)malloc(slm->vocab_size * slm->embedding_dim * sizeof(float));
    float* output_m = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    float* output_v = (float*)malloc(slm->embedding_dim * slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(embed_m, slm->d_embed_m, slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(embed_v, slm->d_embed_v, slm->vocab_size * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(output_m, slm->d_output_m, slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(output_v, slm->d_output_v, slm->embedding_dim * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(embed_m, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(embed_v, sizeof(float), slm->vocab_size * slm->embedding_dim, file);
    fwrite(output_m, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    fwrite(output_v, sizeof(float), slm->embedding_dim * slm->vocab_size, file);
    
    // Free temporary host memory
    free(embeddings); free(output_weights);
    free(embed_m); free(embed_v); free(output_m); free(output_v);
    
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
    int vocab_size, embedding_dim, seq_len, stored_batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize model
    SLM* slm = init_slm(vocab_size, embedding_dim, seq_len, batch_size);
    
    // Allocate temporary host memory
    float* embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* output_weights = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    // Load matrices
    fread(embeddings, sizeof(float), vocab_size * embedding_dim, file);
    fread(output_weights, sizeof(float), embedding_dim * vocab_size, file);
    
    fread(&slm->t, sizeof(int), 1, file);
    
    // Copy matrices to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, embeddings, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_weights, output_weights, embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    float* embed_m = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* embed_v = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* output_m = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    float* output_v = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    fread(embed_m, sizeof(float), vocab_size * embedding_dim, file);
    fread(embed_v, sizeof(float), vocab_size * embedding_dim, file);
    fread(output_m, sizeof(float), embedding_dim * vocab_size, file);
    fread(output_v, sizeof(float), embedding_dim * vocab_size, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(slm->d_embed_m, embed_m, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embed_v, embed_v, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_m, output_m, embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_v, output_v, embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(embeddings); free(output_weights);
    free(embed_m); free(embed_v); free(output_m); free(output_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return slm;
}

#endif