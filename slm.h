#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"

typedef struct {
    SSM** ssms;                 // Dynamic array of state space models
    int num_layers;             // Number of layers
    
    // Layer normalization parameters
    float** d_ln_gamma;         // Layer norm scale parameters for each layer
    float** d_ln_beta;          // Layer norm shift parameters for each layer
    float** d_ln_gamma_grad;    // Gradients for gamma
    float** d_ln_beta_grad;     // Gradients for beta
    float** d_ln_gamma_m;       // Adam momentum for gamma
    float** d_ln_gamma_v;       // Adam velocity for gamma
    float** d_ln_beta_m;        // Adam momentum for beta
    float** d_ln_beta_v;        // Adam velocity for beta
    
    // Layer norm working buffers
    float** d_ln_mean;          // Mean for each layer
    float** d_ln_var;           // Variance for each layer
    float** d_ln_output;        // Normalized output for each layer
    float** d_ln_input_grad;    // Input gradients from layer norm
    
    // Language modeling specific buffers
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    float* d_embeddings_v;      // vocab_size x embed_dim
    
    // Working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float** d_ssm_outputs;      // seq_len x batch_size x embed_dim (for intermediate layers)
    float* d_final_output;      // seq_len x batch_size x vocab_size
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float** d_ssm_gradients;    // seq_len x batch_size x embed_dim (for intermediate layers)
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

// Layer normalization CUDA kernel prototypes
__global__ void layer_norm_forward_kernel(float* output, float* input, float* gamma, float* beta,
                                         float* mean, float* var, int batch_size, int feature_dim, float eps);
__global__ void layer_norm_backward_kernel(float* input_grad, float* gamma_grad, float* beta_grad,
                                          float* output_grad, float* input, float* gamma,
                                          float* mean, float* var, int batch_size, int feature_dim, float eps);
__global__ void adamw_update_kernel_slm(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size);

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

#endif