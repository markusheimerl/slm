#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "../ssm/gpu/fp16/ssm.h"

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
    // Device pointers for layers
    SSM** ssms;                 // Array of state space models
    int num_layers;             // Number of layers
    
    // Device pointers for embeddings
    __half* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;    // vocab_size x embed_dim
    float* d_embeddings_m;       // vocab_size x embed_dim
    float* d_embeddings_v;       // vocab_size x embed_dim
    
    // Device pointers for working buffers
    __half* d_embedded_input;    // seq_len x batch_size x embed_dim
    __half* d_softmax;           // seq_len x batch_size x vocab_size
    __half* d_input_gradients;   // seq_len x batch_size x embed_dim
    float* d_losses;             // seq_len x batch_size
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int vocab_size;
    int embed_dim;
} SLM;

// CUDA kernel prototypes
__global__ void embedding_lookup_kernel(__half* output, __half* embeddings, unsigned char* chars, int batch_size, int embed_dim);
__global__ void softmax_kernel(__half* output, __half* input, int batch_size, int vocab_size);
__global__ void cross_entropy_loss_kernel(float* losses, __half* grad, __half* softmax, unsigned char* targets, int batch_size, int vocab_size);
__global__ void embedding_gradient_kernel(float* embed_grad, __half* input_grad, unsigned char* chars, int batch_size, int embed_dim);

// Function prototypes
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int num_layers, int batch_size);
void free_slm(SLM* slm);
void reset_state_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_X_t, int timestep);
float calculate_loss_slm(SLM* slm, unsigned char* d_y);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_X_t, int timestep);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size);
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature, float top_p);

#endif