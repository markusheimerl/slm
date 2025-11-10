#ifndef SLM_H
#define SLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cblas.h>
#include "transformer/transformer.h"

typedef struct {
    // Token embedding layer
    float* token_embedding;        // [vocab_size x d_model]
    float* token_embedding_grad;   // [vocab_size x d_model]
    
    // Output projection weights
    float* W_output;               // [d_model x vocab_size]
    float* W_output_grad;          // [d_model x vocab_size]
    
    // Adam parameters
    float* token_embedding_m;      // First moment for token embeddings
    float* token_embedding_v;      // Second moment for token embeddings
    float* W_output_m;             // First moment for output weights
    float* W_output_v;             // Second moment for output weights
    float beta1;                   // Exponential decay rate for first moment
    float beta2;                   // Exponential decay rate for second moment
    float epsilon;                 // Small constant for numerical stability
    int t;                         // Time step
    float weight_decay;            // Weight decay parameter for AdamW
    
    // Forward pass buffers
    float* embedded_input;         // [batch_size x seq_len x d_model]
    float* output;                 // [batch_size x seq_len x vocab_size]
    
    // Backward pass buffers
    float* grad_output;            // [batch_size x seq_len x vocab_size]
    
    // Transformer core
    Transformer* transformer;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
    int vocab_size;
} SLM;

// Function prototypes
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size);
void free_slm(SLM* slm);
void forward_pass_slm(SLM* slm, unsigned char* input_tokens);
float calculate_loss_slm(SLM* slm, unsigned char* target_tokens);
void zero_gradients_slm(SLM* slm);
void backward_pass_slm(SLM* slm, unsigned char* input_tokens);
void update_weights_slm(SLM* slm, float learning_rate, int effective_batch_size);
void save_slm(SLM* slm, const char* filename);
SLM* load_slm(const char* filename, int custom_batch_size);

#endif