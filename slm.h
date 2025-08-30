#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "transformer/gpu/transformer.h"
#include "data.h"

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

#define VOCAB_SIZE 256  // Character-level: 0-255

typedef struct {
    Transformer* transformer;
    cublasHandle_t cublas_handle;
    
    // Embedding layers
    float* d_token_embedding;     // [VOCAB_SIZE x d_model]
    float* d_position_embedding;  // [seq_len x d_model]
    float* d_output_projection;   // [d_model x VOCAB_SIZE]
    
    // Embedding gradients
    float* d_token_embedding_grad;
    float* d_position_embedding_grad;
    float* d_output_projection_grad;
    
    // Adam parameters for embeddings
    float* d_token_embedding_m;
    float* d_token_embedding_v;
    float* d_position_embedding_m;
    float* d_position_embedding_v;
    float* d_output_projection_m;
    float* d_output_projection_v;
    
    // Forward pass buffers
    float* d_embedded_input;      // [batch_size x seq_len x d_model]
    float* d_logits;             // [batch_size x seq_len x VOCAB_SIZE]
    float* d_probabilities;      // [batch_size x seq_len x VOCAB_SIZE]
    
    // Backward pass buffers
    float* d_grad_embedded;      // [batch_size x seq_len x d_model]
    float* d_grad_logits;        // [batch_size x seq_len x VOCAB_SIZE]
    
    // Working buffers
    float* d_temp_embedding;     // [batch_size x seq_len x d_model]
    
    // Adam parameters
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Dimensions
    int d_model;
    int seq_len;
    int batch_size;
    int vocab_size;
    int num_layers;
    int mlp_hidden;
} SLM;

// CUDA kernel prototypes
__global__ void add_position_embeddings_kernel(float* embedded, float* pos_emb, int batch_size, int seq_len, int d_model);
__global__ void softmax_kernel_slm(float* probabilities, float* logits, int batch_size, int seq_len, int vocab_size);
__global__ void cross_entropy_backward_kernel(float* grad_logits, float* probabilities, unsigned char* targets, int batch_size, int seq_len, int vocab_size);
__global__ void embedding_backward_kernel(float* embedding_grad, float* grad_embedded, unsigned char* tokens, int batch_size, int seq_len, int d_model, int vocab_size);
__global__ void adamw_update_kernel_slm(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int batch_size);

// Function prototypes
SLM* init_slm(int d_model, int seq_len, int batch_size, int mlp_hidden, int num_layers, cublasHandle_t cublas_handle);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle);
void generate_text(SLM* slm, unsigned char* seed, int seed_len, int generate_len, float temperature);

#endif