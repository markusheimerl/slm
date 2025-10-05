#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include "../transformer/gpu/transformer.h"
#include "../transformer/mlp/gpu/mlp.h"

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

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Token embedding layer
    float* d_token_embedding;      // [vocab_size x d_model]
    float* d_token_embedding_grad; // [vocab_size x d_model]
    
    // Adam parameters
    float* d_token_embedding_m;    // First moment for token embeddings
    float* d_token_embedding_v;    // Second moment for token embeddings
    float beta1;                   // Exponential decay rate for first moment
    float beta2;                   // Exponential decay rate for second moment
    float epsilon;                 // Small constant for numerical stability
    int t;                         // Time step
    float weight_decay;            // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* d_embedded_input;       // [batch_size x seq_len x d_model]

    // Loss computation buffer
    float* d_loss_result;          // [1]
    
    // Transformer core
    Transformer* transformer;
    
    // Output projection MLP
    MLP* output_mlp;
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
    int vocab_size;
} SLM;

// Function prototypes
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif