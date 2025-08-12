#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ssm/gpu/ssm.h"

typedef struct {
    // Device pointers for layers
    SSM** ssms;                 // Array of state space models
    int num_layers;             // Number of layers
    
    // Device pointers for embeddings
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    
    // Device pointers for working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float* d_losses;            // seq_len x batch_size
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int vocab_size;
    int embed_dim;
} SLM;

// CUDA kernel prototypes
__global__ void embedding_lookup_kernel(float* output, float* embeddings, unsigned char* chars, int batch_size, int embed_dim);
__global__ void softmax_kernel(float* output, float* input, int batch_size, int vocab_size);
__global__ void cross_entropy_loss_kernel(float* losses, float* grad, float* softmax, unsigned char* targets, int batch_size, int vocab_size);
__global__ void embedding_gradient_kernel(float* embed_grad, float* input_grad, unsigned char* chars, int batch_size, int embed_dim);

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