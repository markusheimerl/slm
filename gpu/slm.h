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

// cuBLASLt matrix multiplication macro
#ifndef LT_MATMUL
#define LT_MATMUL(slm, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(slm->cublaslt_handle, slm->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)
#endif

typedef struct {
    // Token embedding layer
    float* d_token_embedding;      // [vocab_size x d_model]
    float* d_token_embedding_grad; // [vocab_size x d_model]
    
    // Output projection weights
    float* d_W_output;             // [d_model x vocab_size]
    float* d_W_output_grad;        // [d_model x vocab_size]
    
    // Adam parameters
    float* d_token_embedding_m;    // First moment for token embeddings
    float* d_token_embedding_v;    // Second moment for token embeddings
    float* d_W_output_m;           // First moment for output weights
    float* d_W_output_v;           // Second moment for output weights
    float beta1;                   // Exponential decay rate for first moment
    float beta2;                   // Exponential decay rate for second moment
    float epsilon;                 // Small constant for numerical stability
    int t;                         // Time step
    float weight_decay;            // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* d_embedded_input;       // [batch_size x seq_len x d_model]
    float* d_output;               // [batch_size x seq_len x vocab_size]
    
    // Backward pass buffers
    float* d_grad_output;          // [batch_size x seq_len x vocab_size]

    // Loss computation buffer
    float* d_loss_result;          // [1]
    
    // Transformer core
    Transformer* transformer;
    
    // cuBLASLt handle and descriptor
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    
    // Matrix layouts
    cublasLtMatrixLayout_t output_weight_layout;  // [d_model x vocab_size]
    cublasLtMatrixLayout_t seq_flat_d_model_layout;   // [batch_size * seq_len x d_model]
    cublasLtMatrixLayout_t seq_flat_vocab_layout; // [batch_size * seq_len x vocab_size]
    
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
void forward_pass_slm(SLM* slm, unsigned short* d_input_tokens);
float calculate_loss_slm(SLM* slm, unsigned short* d_target_tokens);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned short* d_input_tokens);
void update_weights_slm(SLM* slm, float learning_rate, int effective_batch_size);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif