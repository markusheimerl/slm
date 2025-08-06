#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"
#include "mlp/gpu/mlp.h"

// LayerNorm structure
typedef struct {
    // Device pointers for learnable parameters
    float* d_gamma;      // scale parameter: embed_dim
    float* d_beta;       // bias parameter: embed_dim
    float* d_gamma_grad; // gradients for gamma
    float* d_beta_grad;  // gradients for beta
    
    // Device pointers for Adam parameters
    float* d_gamma_m;    // First moment for gamma
    float* d_gamma_v;    // Second moment for gamma  
    float* d_beta_m;     // First moment for beta
    float* d_beta_v;     // Second moment for beta
    
    // Working buffers
    float* d_normalized; // normalized output: seq_len x batch_size x embed_dim
    float* d_mean;       // mean: seq_len x batch_size
    float* d_variance;   // variance: seq_len x batch_size
    
    // Dimensions
    int embed_dim;
    int seq_len;
    int batch_size;
    float epsilon;       // small constant for numerical stability
} LayerNorm;

typedef struct {
    SSM** ssms;                 // Dynamic array of state space models
    MLP** mlps;                 // Dynamic array of MLPs
    LayerNorm** layer_norms;    // Dynamic array of LayerNorms (one per layer)
    LayerNorm* pre_norm;        // LayerNorm after embeddings
    int num_layers;             // Number of layers
    
    // Language modeling specific buffers
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    float* d_embeddings_v;      // vocab_size x embed_dim
    
    // Working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float** d_ssm_outputs;      // seq_len x batch_size x embed_dim
    float** d_mlp_outputs;      // seq_len x batch_size x embed_dim
    float** d_layernorm_outputs; // seq_len x batch_size x embed_dim (after each layer)
    float* d_pre_norm_output;   // seq_len x batch_size x embed_dim (after embedding LayerNorm)
    float* d_final_output;      // seq_len x batch_size x vocab_size
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float** d_ssm_gradients;    // seq_len x batch_size x embed_dim
    float** d_mlp_gradients;    // seq_len x batch_size x embed_dim
    float** d_layernorm_gradients; // seq_len x batch_size x embed_dim
    float* d_pre_norm_gradients; // seq_len x batch_size x embed_dim
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

// LayerNorm CUDA kernel prototypes
__global__ void layernorm_forward_kernel(float* output, float* input, float* gamma, float* beta,
                                       float* mean, float* variance, int seq_len, int batch_size, 
                                       int embed_dim, float epsilon);
__global__ void layernorm_backward_kernel(float* grad_input, float* grad_gamma, float* grad_beta,
                                        float* grad_output, float* input, float* gamma, 
                                        float* mean, float* variance, int seq_len, int batch_size,
                                        int embed_dim, float epsilon);
__global__ void adamw_update_layernorm_kernel(float* param, float* grad, float* m, float* v,
                                            float beta1, float beta2, float epsilon, float lr,
                                            float weight_decay, float alpha_t, int size);

// Function prototypes
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int num_layers, int batch_size);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* d_X);
float calculate_loss_slm(SLM* slm, unsigned char* d_y);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* d_X);
void update_weights_slm(SLM* slm, float learning_rate);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size);
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature, float top_p);

// LayerNorm function prototypes
LayerNorm* init_layernorm(int embed_dim, int seq_len, int batch_size);
void free_layernorm(LayerNorm* ln);
void forward_layernorm(LayerNorm* ln, float* d_input, float* d_output);
void backward_layernorm(LayerNorm* ln, float* d_grad_output, float* d_grad_input);
void zero_gradients_layernorm(LayerNorm* ln);
void update_weights_layernorm(LayerNorm* ln, float learning_rate, float beta1, float beta2, int timestep);

#endif