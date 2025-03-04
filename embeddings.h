#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Error checking macro for CUDA calls (if not already defined)
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

// ---------------------------------------------------------------------
// Structure for holding embeddings and their gradients
// ---------------------------------------------------------------------
typedef struct {
    // Embeddings (device and host memory)
    float* d_embeddings;      // vocab_size x embedding_dim
    float* h_embeddings;      // vocab_size x embedding_dim
    
    // Gradients (device memory)
    float* d_embedding_grads; // vocab_size x embedding_dim
    
    // Adam optimizer first (m) and second (v) moment estimates (device pointers)
    float* d_embedding_m;     // First moment
    float* d_embedding_v;     // Second moment
    
    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
} Embeddings;

// ---------------------------------------------------------------------
// CUDA kernel: Embed input bytes (forward pass)
// ---------------------------------------------------------------------
__global__ void embed_bytes_kernel(float* output, 
                                  const unsigned char* bytes, 
                                  const float* embeddings, 
                                  int batch_size, 
                                  int embedding_dim) {
    int batch_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (batch_idx < batch_size && emb_idx < embedding_dim) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding table
        int embedding_offset = byte_val * embedding_dim;
        
        // Copy the embedding vector to output
        output[batch_idx * embedding_dim + emb_idx] = embeddings[embedding_offset + emb_idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Compute embedding gradients (backward pass)
// ---------------------------------------------------------------------
__global__ void embedding_gradient_kernel(float* embedding_grads,
                                         const float* input_grads,
                                         const unsigned char* bytes,
                                         int batch_size,
                                         int embedding_dim) {
    int batch_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (batch_idx < batch_size && emb_idx < embedding_dim) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding gradient table
        int grad_offset = byte_val * embedding_dim + emb_idx;
        
        // Add gradient to embedding gradient table
        atomicAdd(&embedding_grads[grad_offset], 
                 input_grads[batch_idx * embedding_dim + emb_idx]);
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: AdamW update for embeddings
// ---------------------------------------------------------------------
__global__ void adamw_embeddings_kernel(float* W, const float* grad, float* m, float* v, 
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
// Function: Initialize embeddings
// Initializes the embeddings structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
Embeddings* init_embeddings(int vocab_size, int embedding_dim) {
    Embeddings* emb = (Embeddings*)malloc(sizeof(Embeddings));
    emb->vocab_size = vocab_size;
    emb->embedding_dim = embedding_dim;
    
    // Set Adam hyperparameters
    emb->beta1 = 0.9f;
    emb->beta2 = 0.999f;
    emb->epsilon = 1e-8f;
    emb->weight_decay = 0.01f;
    emb->adam_t = 0;
    
    // Allocate host memory for embeddings
    emb->h_embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    
    // Initialize matrices with scaled random values
    float scale = 1.0f / sqrtf(embedding_dim);
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        emb->h_embeddings[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    
    // Allocate device memory for embeddings
    CHECK_CUDA(cudaMalloc(&emb->d_embeddings, vocab_size * embedding_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_grads, vocab_size * embedding_dim * sizeof(float)));
    
    // Allocate device memory for Adam first and second moment estimates and initialize to zero
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_m, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_v, vocab_size * embedding_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMemset(emb->d_embedding_grads, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(emb->d_embedding_m, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(emb->d_embedding_v, 0, vocab_size * embedding_dim * sizeof(float)));
    
    // Copy embeddings from host to device
    CHECK_CUDA(cudaMemcpy(emb->d_embeddings, emb->h_embeddings, 
                         vocab_size * embedding_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    return emb;
}

// ---------------------------------------------------------------------
// Function: Forward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_forward(Embeddings* emb, unsigned char* d_bytes, float* d_output, int batch_size) {
    embed_bytes_kernel<<<batch_size, emb->embedding_dim>>>(
        d_output, d_bytes, emb->d_embeddings, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Backward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_backward(Embeddings* emb, float* d_input_grads, unsigned char* d_bytes, int batch_size) {
    embedding_gradient_kernel<<<batch_size, emb->embedding_dim>>>(
        emb->d_embedding_grads, d_input_grads, d_bytes, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Zero gradients for embeddings
// ---------------------------------------------------------------------
void zero_embedding_gradients(Embeddings* emb) {
    size_t emb_size = emb->vocab_size * emb->embedding_dim * sizeof(float);
    CHECK_CUDA(cudaMemset(emb->d_embedding_grads, 0, emb_size));
}

// ---------------------------------------------------------------------
// Function: Update embeddings with AdamW optimizer
// ---------------------------------------------------------------------
void update_embeddings(Embeddings* emb, float learning_rate, int batch_size) {
    emb->adam_t++; // Increment time step
    
    float bias_correction1 = 1.0f - powf(emb->beta1, (float)emb->adam_t);
    float bias_correction2 = 1.0f - powf(emb->beta2, (float)emb->adam_t);
    
    int block_size = 256;
    int size = emb->vocab_size * emb->embedding_dim;
    int num_blocks = (size + block_size - 1) / block_size;
    
    adamw_embeddings_kernel<<<num_blocks, block_size>>>(
        emb->d_embeddings, emb->d_embedding_grads, emb->d_embedding_m, emb->d_embedding_v,
        size, emb->beta1, emb->beta2, emb->epsilon, emb->weight_decay, 
        learning_rate, batch_size, bias_correction1, bias_correction2);
}

// ---------------------------------------------------------------------
// Function: Free embeddings
// Frees all allocated memory (both device and host).
// ---------------------------------------------------------------------
void free_embeddings(Embeddings* emb) {
    if (!emb) return;
    
    // Free device memory
    cudaFree(emb->d_embeddings);
    cudaFree(emb->d_embedding_grads);
    cudaFree(emb->d_embedding_m);
    cudaFree(emb->d_embedding_v);
    
    // Free host memory
    free(emb->h_embeddings);
    free(emb);
}

// ---------------------------------------------------------------------
// Function: Save embeddings
// Saves the embeddings to a binary file.
// ---------------------------------------------------------------------
void save_embeddings(Embeddings* emb, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write dimensions
    fwrite(&emb->vocab_size, sizeof(int), 1, file);
    fwrite(&emb->embedding_dim, sizeof(int), 1, file);
    
    // Write Adam hyperparameters
    fwrite(&emb->beta1, sizeof(float), 1, file);
    fwrite(&emb->beta2, sizeof(float), 1, file);
    fwrite(&emb->epsilon, sizeof(float), 1, file);
    fwrite(&emb->weight_decay, sizeof(float), 1, file);
    fwrite(&emb->adam_t, sizeof(int), 1, file);
    
    size_t size = emb->vocab_size * emb->embedding_dim * sizeof(float);
    
    // Allocate host memory for copying from device
    float* h_embeddings = (float*)malloc(size);
    float* h_embedding_m = (float*)malloc(size);
    float* h_embedding_v = (float*)malloc(size);
    
    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_embeddings, emb->d_embeddings, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embedding_m, emb->d_embedding_m, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embedding_v, emb->d_embedding_v, size, cudaMemcpyDeviceToHost));
    
    // Write data to file
    fwrite(h_embeddings, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    fwrite(h_embedding_m, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    fwrite(h_embedding_v, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    
    // Free host memory
    free(h_embeddings);
    free(h_embedding_m);
    free(h_embedding_v);
    
    fclose(file);
    printf("Embeddings saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: Load embeddings
// Loads the embeddings from a binary file and initializes.
// ---------------------------------------------------------------------
Embeddings* load_embeddings(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int vocab_size, embedding_dim;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    
    Embeddings* emb = init_embeddings(vocab_size, embedding_dim);
    
    // Read Adam hyperparameters
    fread(&emb->beta1, sizeof(float), 1, file);
    fread(&emb->beta2, sizeof(float), 1, file);
    fread(&emb->epsilon, sizeof(float), 1, file);
    fread(&emb->weight_decay, sizeof(float), 1, file);
    fread(&emb->adam_t, sizeof(int), 1, file);
    
    size_t size = vocab_size * embedding_dim * sizeof(float);
    
    // Allocate host buffers for reading
    float* h_embeddings = (float*)malloc(size);
    float* h_embedding_m = (float*)malloc(size);
    float* h_embedding_v = (float*)malloc(size);
    
    // Read data from file to host memory
    fread(h_embeddings, sizeof(float), vocab_size * embedding_dim, file);
    fread(h_embedding_m, sizeof(float), vocab_size * embedding_dim, file);
    fread(h_embedding_v, sizeof(float), vocab_size * embedding_dim, file);
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(emb->d_embeddings, h_embeddings, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(emb->d_embedding_m, h_embedding_m, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(emb->d_embedding_v, h_embedding_v, size, cudaMemcpyHostToDevice));
    
    // Update host copy
    memcpy(emb->h_embeddings, h_embeddings, size);
    
    // Free temporary host memory
    free(h_embeddings);
    free(h_embedding_m);
    free(h_embedding_v);
    
    fclose(file);
    printf("Embeddings loaded from %s\n", filename);
    return emb;
}

// ---------------------------------------------------------------------
// CUDA kernel: Softmax for probabilities output
// ---------------------------------------------------------------------
__global__ void softmax_kernel(float* logits, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointer to this batch item's prediction vector
        float* batch_logits = logits + batch_idx * vocab_size;
        
        // Find max value for numerical stability
        float max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
        
        // Compute exp(logits - max) and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] = expf(batch_logits[i] - max_val);
            sum_exp += batch_logits[i];
        }
        
        // Ensure sum is not zero
        sum_exp = fmaxf(sum_exp, 1e-10f);
        
        // Normalize to get probabilities
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] /= sum_exp;
        }
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Cross-entropy loss computation
// ---------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(float* loss, 
                                         float* d_error, 
                                         const float* probs, 
                                         const float* targets, 
                                         int batch_size, 
                                         int vocab_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* batch_probs = probs + batch_idx * vocab_size;
        const float* batch_target = targets + batch_idx * vocab_size;
        float* batch_error = d_error + batch_idx * vocab_size;
        float batch_loss = 0.0f;
        
        for (int i = 0; i < vocab_size; i++) {
            float prob = fmaxf(fminf(batch_probs[i], 1.0f - 1e-7f), 1e-7f);
            if (batch_target[i] > 0.0f) {
                batch_loss -= batch_target[i] * logf(prob);
            }
            
            // Gradient for softmax with cross-entropy: prob - target
            batch_error[i] = batch_probs[i] - batch_target[i];
        }
        
        atomicAdd(loss, batch_loss);
    }
}

// ---------------------------------------------------------------------
// Function: Calculate cross-entropy loss
// ---------------------------------------------------------------------
float calculate_cross_entropy_loss(SSM* ssm, float* d_y) {
    // Apply softmax to get probabilities
    softmax_kernel<<<ssm->batch_size, 1>>>(ssm->d_predictions, 
                                          ssm->batch_size, 
                                          ssm->output_dim);
    
    // Initialize loss to zero
    float h_loss = 0.0f;
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    // Compute cross-entropy loss and gradients
    cross_entropy_loss_kernel<<<ssm->batch_size, 1>>>(d_loss, 
                                                     ssm->d_error, 
                                                     ssm->d_predictions, 
                                                     d_y, 
                                                     ssm->batch_size, 
                                                     ssm->output_dim);
    
    // Copy loss back to host
    CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_loss));
    
    // Return average loss per batch item
    return h_loss / ssm->batch_size;
}

#endif // EMBEDDINGS_H