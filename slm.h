#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"
#include "mlp/gpu/mlp.h"

typedef struct {
    SSM* ssm1;                  // First state space model layer
    SSM* ssm2;                  // Second state space model layer
    MLP* mlp;                   // Multi-layer perceptron for output mapping
    
    // Language modeling specific buffers
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    float* d_embeddings_v;      // vocab_size x embed_dim
    
    // Working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float* d_ssm1_output;       // seq_len x batch_size x embed_dim (output from first SSM)
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float* d_ssm1_gradients;    // seq_len x batch_size x embed_dim (gradients for first SSM input)
    float* d_losses;            // seq_len x batch_size
    
    // Dimensions
    int vocab_size;
    int embed_dim;
} SLM;

// CUDA kernel prototypes
__global__ void embedding_lookup_kernel(float* output, float* embeddings, unsigned char* chars, 
                                       int batch_size, int embed_dim);
__global__ void softmax_kernel(float* output, float* input, int batch_size, int vocab_size);
__global__ void cross_entropy_gradient_kernel(float* grad, float* softmax, unsigned char* targets, 
                                             int batch_size, int vocab_size);
__global__ void cross_entropy_loss_kernel(float* losses, float* softmax, unsigned char* targets, 
                                         int batch_size, int vocab_size);
__global__ void embedding_gradient_kernel(float* embed_grad, float* input_grad, unsigned char* chars,
                                         int batch_size, int embed_dim);

// Function prototypes
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int batch_size);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_X);
float calculate_loss_slm(SLM* slm, unsigned char* d_y);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_X);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size);
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature);

#endif