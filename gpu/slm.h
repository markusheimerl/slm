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

typedef struct {
    // Embedding layers
    float* d_token_embedding;      // [vocab_size x d_model]
    float* d_position_embedding;   // [seq_len x d_model]
    float* d_output_projection;    // [d_model x vocab_size]
    
    // Gradients for embedding layers
    float* d_token_embedding_grad; // [vocab_size x d_model]
    float* d_position_embedding_grad; // [seq_len x d_model]
    float* d_output_projection_grad;  // [d_model x vocab_size]
    
    // Adam parameters for embeddings
    float* d_token_embedding_m, *d_token_embedding_v;
    float* d_position_embedding_m, *d_position_embedding_v;
    float* d_output_projection_m, *d_output_projection_v;
    float beta1, beta2, epsilon, weight_decay;
    int t;
    
    // Forward pass buffers
    float* d_embedded_input;    // [batch_size x seq_len x d_model]
    float* d_logits;           // [batch_size x seq_len x vocab_size]
    float* d_probs;            // [batch_size x seq_len x vocab_size]
    
    // Backward pass buffers
    float* d_grad_embedded;    // [batch_size x seq_len x d_model]
    float* d_grad_logits;      // [batch_size x seq_len x vocab_size]
    
    // Loss computation buffer
    float* d_loss_result;      // [1]
    
    // Transformer core
    Transformer* transformer;
    
    // cuBLASLt handle and layouts
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDesc_t matmul_NT_desc;
    cublasLtMatmulDesc_t matmul_TN_desc;
    
    // Matrix layouts
    cublasLtMatrixLayout_t token_emb_layout;      // [vocab_size x d_model]
    cublasLtMatrixLayout_t pos_emb_layout;        // [seq_len x d_model]
    cublasLtMatrixLayout_t output_proj_layout;    // [d_model x vocab_size]
    cublasLtMatrixLayout_t embedded_layout;       // [batch_size x seq_len x d_model]
    cublasLtMatrixLayout_t logits_layout;         // [batch_size x seq_len x vocab_size]
    
    // Gradient computation layouts
    cublasLtMatrixLayout_t transformer_out_for_grad_layout;
    cublasLtMatrixLayout_t grad_logits_for_grad_layout;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
    int vocab_size;
} SLM;

// Function prototypes
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif