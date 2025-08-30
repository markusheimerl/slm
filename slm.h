#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "transformer/gpu/transformer.h"

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
    // Transformer component
    Transformer* transformer;
    
    // Device pointers for embeddings
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // First moment for embeddings (AdamW)
    float* d_embeddings_v;      // Second moment for embeddings (AdamW)
    
    // AdamW parameters for embeddings
    float beta1;
    float beta2;
    float epsilon;
    int t;                      // Time step
    float weight_decay;
    
    // Device pointers for working buffers
    float* d_embedded_input;    // batch_size x seq_len x embed_dim
    float* d_softmax;           // batch_size x seq_len x vocab_size
    float* d_input_gradients;   // batch_size x seq_len x embed_dim
    float* d_losses;            // batch_size x seq_len
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int vocab_size;
    int embed_dim;
    int seq_len;
    int batch_size;
} SLM;

// CUDA kernel prototypes
__global__ void embedding_lookup_kernel(float* output, float* embeddings, unsigned char* chars, 
                                       int batch_size, int seq_len, int embed_dim);
__global__ void softmax_kernel(float* output, float* input, int batch_size, int seq_len, int vocab_size);
__global__ void cross_entropy_loss_kernel(float* losses, float* grad, float* softmax, unsigned char* targets, 
                                         int batch_size, int seq_len, int vocab_size);
__global__ void embedding_gradient_kernel(float* embed_grad, float* input_grad, unsigned char* chars,
                                         int batch_size, int seq_len, int embed_dim);
__global__ void adamw_update_kernel_embeddings(float* weight, float* grad, float* m, float* v,
                                              float beta1, float beta2, float epsilon, float learning_rate,
                                              float weight_decay, float alpha_t, int size, int total_samples);

// Function prototypes
SLM* init_slm(int embed_dim, int seq_len, int num_layers, int mlp_hidden, int batch_size);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_X);
float calculate_loss_slm(SLM* slm, unsigned char* d_y);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_X);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size);
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature, float top_p);

#endif