#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
    // Device pointers for state space matrices
    float* d_A;           // state_dim x state_dim (state transition)
    float* d_B;           // state_dim x input_dim (input to state)
    float* d_C;           // output_dim x state_dim (state to output)
    float* d_D;           // output_dim x input_dim (input to output)
    
    // Device pointers for gradients
    float* d_A_grad;      // state_dim x state_dim
    float* d_B_grad;      // state_dim x input_dim
    float* d_C_grad;      // output_dim x state_dim
    float* d_D_grad;      // output_dim x input_dim
    
    // Device pointers for Adam parameters
    float* d_A_m; float* d_A_v;
    float* d_B_m; float* d_B_v;
    float* d_C_m; float* d_C_v;
    float* d_D_m; float* d_D_v;
    
    float beta1, beta2, epsilon;
    int t;
    float weight_decay;
    
    // Device pointers for helper arrays (time-major format)
    float* d_states;          // seq_len x batch_size x state_dim
    float* d_predictions;     // seq_len x batch_size x output_dim (logits)
    float* d_softmax;         // seq_len x batch_size x output_dim (probabilities)
    float* d_error;          // seq_len x batch_size x output_dim
    float* d_state_error;    // seq_len x batch_size x state_dim
    float* d_state_outputs;  // seq_len x batch_size x state_dim
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int state_dim;
    int output_dim;
    int seq_len;
    int batch_size;
} SLM;

// Initialize the state space language model
SLM* init_slm(int input_dim, int state_dim, int output_dim, int seq_len, int batch_size);

// Free memory
void free_slm(SLM* slm);

// Forward pass
void forward_pass_slm(SLM* slm, float* d_X);

// Calculate cross-entropy loss
float calculate_loss_slm(SLM* slm, float* d_y);

// Zero gradients
void zero_gradients_slm(SLM* slm);

// Backward pass
void backward_pass_slm(SLM* slm, float* d_X);

// Update weights using AdamW
void update_weights_slm(SLM* slm, float learning_rate);

// Save model
void save_slm(SLM* slm, const char* filename);

// Load model
SLM* load_slm(const char* filename, int custom_batch_size);

#endif