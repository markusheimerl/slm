#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

#define MAX_SEQ_LENGTH 1024
#define MAX_VOCAB_SIZE 256
#define EMBED_DIM 512
#define NUM_LAYERS 8
#define BATCH_SIZE 64
#define TEMPERATURE 0.7f

// MixerBlock structure
typedef struct {
    // Token mixing parameters
    float* token_mixing_weight;         // seq_length x seq_length
    float* token_mixing_weight_grad;    // seq_length x seq_length
    float* token_mixing_mask;           // seq_length x seq_length (lower triangular mask)
    float* token_mixing_m;              // Adam first moment
    float* token_mixing_v;              // Adam second moment
    
    // Channel mixing parameters
    float* channel_mixing_weight;       // embed_dim x embed_dim
    float* channel_mixing_weight_grad;  // embed_dim x embed_dim
    float* channel_mixing_m;            // Adam first moment
    float* channel_mixing_v;            // Adam second moment
    
    // Forward pass buffers
    float* input_buffer;                // Input to the block
    float* residual;                    // Residual connection
    float* transposed;                  // For token mixing
    float* token_mixed;                 // After token mixing
    float* token_mix_activated;         // After activation
    float* channel_mixed;               // After channel mixing
    float* channel_mix_activated;       // After activation
    
    // Backward pass buffers
    float* d_output;                    // Gradient from next layer
    float* d_token_mixed;               // Gradient for token mixing
    float* d_channel_mixed;             // Gradient for channel mixing
    float* d_input;                     // Gradient to pass to previous layer
    
    // Additional preallocated buffers
    float* masked_weights;              // For token mixing
    float* d_channel_activated;         // For backward pass
    float* d_output_transposed;         // For token mixing backward
    float* d_input_transposed;          // For token mixing backward
    float* temp_grad;                   // For token mixing backward
    
    // Dimensions
    int embed_dim;
    int seq_length;
} MixerBlock;

// MixerModel structure
typedef struct {
    // Embedding parameters
    float* embedding_weight;            // vocab_size x embed_dim
    float* embedding_weight_grad;       // vocab_size x embed_dim
    float* embedding_m;                 // Adam first moment
    float* embedding_v;                 // Adam second moment
    
    // Output projection parameters
    float* out_proj_weight;             // vocab_size x embed_dim
    float* out_proj_weight_grad;        // vocab_size x embed_dim
    float* out_proj_m;                  // Adam first moment
    float* out_proj_v;                  // Adam second moment
    
    // Array of MixerBlocks
    MixerBlock** blocks;
    
    // Forward pass buffers
    float* embeddings;                  // Initial embeddings
    float* block_outputs;               // Outputs from each block
    float* logits;                      // Final logits
    
    // Backward pass buffers
    float* d_logits;                    // Gradient of loss w.r.t. logits
    float* d_block_outputs;             // Gradient of loss w.r.t. block outputs
    
    // Additional preallocated buffers
    float* softmax_probs;               // For loss calculation
    
    // Adam optimizer parameters
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;                              // Time step counter
    
    // Dimensions and hyperparameters
    int vocab_size;
    int embed_dim;
    int num_layers;
    int seq_length;
    int batch_size;
} MixerModel;

// Helper function for sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Helper function for SiLU (Swish) activation: x * sigmoid(x)
float silu(float x) {
    return x * sigmoid(x);
}

// Helper function for SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
float silu_derivative(float x) {
    float sig_x = sigmoid(x);
    return sig_x + x * sig_x * (1.0f - sig_x);
}

// Initialize a MixerBlock
MixerBlock* init_mixer_block(int embed_dim, int seq_length) {
    MixerBlock* block = (MixerBlock*)malloc(sizeof(MixerBlock));
    
    block->embed_dim = embed_dim;
    block->seq_length = seq_length;
    
    // Initialize token mixing weights with scaled random values
    block->token_mixing_weight = (float*)malloc(seq_length * seq_length * sizeof(float));
    block->token_mixing_weight_grad = (float*)calloc(seq_length * seq_length, sizeof(float));
    block->token_mixing_m = (float*)calloc(seq_length * seq_length, sizeof(float));
    block->token_mixing_v = (float*)calloc(seq_length * seq_length, sizeof(float));
    
    float scale_token = 1.0f / sqrtf((float)seq_length);
    for (int i = 0; i < seq_length * seq_length; i++) {
        block->token_mixing_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_token;
    }
    
    // Initialize token mixing mask (lower triangular for causal attention)
    block->token_mixing_mask = (float*)malloc(seq_length * seq_length * sizeof(float));
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            block->token_mixing_mask[i * seq_length + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize channel mixing weights
    block->channel_mixing_weight = (float*)malloc(embed_dim * embed_dim * sizeof(float));
    block->channel_mixing_weight_grad = (float*)calloc(embed_dim * embed_dim, sizeof(float));
    block->channel_mixing_m = (float*)calloc(embed_dim * embed_dim, sizeof(float));
    block->channel_mixing_v = (float*)calloc(embed_dim * embed_dim, sizeof(float));
    
    float scale_channel = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < embed_dim * embed_dim; i++) {
        block->channel_mixing_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_channel;
    }
    
    // Allocate forward pass buffers
    block->input_buffer = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->residual = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->transposed = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->token_mixed = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->token_mix_activated = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->channel_mixed = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->channel_mix_activated = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    
    // Allocate backward pass buffers
    block->d_output = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->d_token_mixed = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->d_channel_mixed = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->d_input = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    
    // Allocate additional preallocated buffers
    block->masked_weights = (float*)malloc(seq_length * seq_length * sizeof(float));
    block->d_channel_activated = (float*)malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    block->d_output_transposed = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->d_input_transposed = (float*)malloc(BATCH_SIZE * embed_dim * seq_length * sizeof(float));
    block->temp_grad = (float*)calloc(seq_length * seq_length, sizeof(float));
    
    return block;
}

void free_mixer_block(MixerBlock* block) {
    // Free weights and gradients
    free(block->token_mixing_weight);
    free(block->token_mixing_weight_grad);
    free(block->token_mixing_mask);
    free(block->token_mixing_m);
    free(block->token_mixing_v);
    
    free(block->channel_mixing_weight);
    free(block->channel_mixing_weight_grad);
    free(block->channel_mixing_m);
    free(block->channel_mixing_v);
    
    // Free forward pass buffers
    free(block->input_buffer);
    free(block->residual);
    free(block->transposed);
    free(block->token_mixed);
    free(block->token_mix_activated);
    free(block->channel_mixed);
    free(block->channel_mix_activated);
    
    // Free backward pass buffers
    free(block->d_output);
    free(block->d_token_mixed);
    free(block->d_channel_mixed);
    free(block->d_input);
    
    // Free additional preallocated buffers
    free(block->masked_weights);
    free(block->d_channel_activated);
    free(block->d_output_transposed);
    free(block->d_input_transposed);
    free(block->temp_grad);
    
    free(block);
}

MixerModel* init_mixer_model(int vocab_size, int embed_dim, int num_layers, int seq_length, int batch_size) {
    MixerModel* model = (MixerModel*)malloc(sizeof(MixerModel));
    
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->num_layers = num_layers;
    model->seq_length = seq_length;
    model->batch_size = batch_size;
    
    // Initialize Adam parameters
    model->beta1 = 0.9f;
    model->beta2 = 0.999f;
    model->epsilon = 1e-8f;
    model->weight_decay = 0.01f;
    model->t = 0;
    
    // Initialize embedding weights
    model->embedding_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    model->embedding_weight_grad = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    model->embedding_m = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    model->embedding_v = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    
    float scale_embed = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        model->embedding_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_embed;
    }
    
    // Initialize output projection weights
    model->out_proj_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    model->out_proj_weight_grad = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    model->out_proj_m = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    model->out_proj_v = (float*)calloc(vocab_size * embed_dim, sizeof(float));
    
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        model->out_proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_embed;
    }
    
    // Initialize MixerBlocks
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_mixer_block(embed_dim, seq_length);
    }
    
    // Allocate forward pass buffers
    model->embeddings = (float*)malloc(batch_size * seq_length * embed_dim * sizeof(float));
    model->block_outputs = (float*)malloc((num_layers + 1) * batch_size * seq_length * embed_dim * sizeof(float));
    model->logits = (float*)malloc(batch_size * seq_length * vocab_size * sizeof(float));
    
    // Allocate backward pass buffers
    model->d_logits = (float*)malloc(batch_size * seq_length * vocab_size * sizeof(float));
    model->d_block_outputs = (float*)malloc((num_layers + 1) * batch_size * seq_length * embed_dim * sizeof(float));
    
    // Allocate additional preallocated buffers
    model->softmax_probs = (float*)malloc(vocab_size * sizeof(float));
    
    return model;
}

void free_mixer_model(MixerModel* model) {
    // Free weights and gradients
    free(model->embedding_weight);
    free(model->embedding_weight_grad);
    free(model->embedding_m);
    free(model->embedding_v);
    
    free(model->out_proj_weight);
    free(model->out_proj_weight_grad);
    free(model->out_proj_m);
    free(model->out_proj_v);
    
    // Free MixerBlocks
    for (int i = 0; i < model->num_layers; i++) {
        free_mixer_block(model->blocks[i]);
    }
    free(model->blocks);
    
    // Free forward pass buffers
    free(model->embeddings);
    free(model->block_outputs);
    free(model->logits);
    
    // Free backward pass buffers
    free(model->d_logits);
    free(model->d_block_outputs);
    
    // Free additional preallocated buffers
    free(model->softmax_probs);
    
    free(model);
}

// Apply embedding layer
void apply_embedding(MixerModel* model, int* input_tokens) {
    for (int b = 0; b < model->batch_size; b++) {
        for (int s = 0; s < model->seq_length; s++) {
            int token_idx = input_tokens[b * model->seq_length + s];
            for (int e = 0; e < model->embed_dim; e++) {
                model->embeddings[b * model->seq_length * model->embed_dim + s * model->embed_dim + e] = 
                    model->embedding_weight[token_idx * model->embed_dim + e];
            }
        }
    }
    
    // Copy embeddings to the first block output buffer
    memcpy(&model->block_outputs[0], model->embeddings, 
           model->batch_size * model->seq_length * model->embed_dim * sizeof(float));
}

// Forward pass through a MixerBlock with optimized CBLAS for both token and channel mixing
void mixer_block_forward(MixerBlock* block, float* input, float* output, int batch_size) {
    // Store input for backward pass
    memcpy(block->input_buffer, input, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Token Mixing -----------------------------------------
    
    // Save input for residual connection
    memcpy(block->residual, input, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Transpose from [batch, seq, embed] to [batch, embed, seq] for token mixing
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < block->seq_length; s++) {
            for (int e = 0; e < block->embed_dim; e++) {
                block->transposed[b * block->embed_dim * block->seq_length + e * block->seq_length + s] = 
                    input[b * block->seq_length * block->embed_dim + s * block->embed_dim + e];
            }
        }
    }
    
    // Create masked token mixing weights
    for (int i = 0; i < block->seq_length; i++) {
        for (int j = 0; j < block->seq_length; j++) {
            block->masked_weights[i * block->seq_length + j] = 
                block->token_mixing_weight[i * block->seq_length + j] * 
                block->token_mixing_mask[i * block->seq_length + j];
        }
    }
    
    // Token mixing - using CBLAS GEMM
    int combined_batch_embed = batch_size * block->embed_dim;
    
    // Using GEMM: token_mixed = transposed * masked_weights^T
    cblas_sgemm(CblasRowMajor,          // Layout (row-major)
                CblasNoTrans,           // No transpose for input
                CblasTrans,             // Transpose weight matrix
                combined_batch_embed,   // Number of rows in input (batch * embed_dim)
                block->seq_length,      // Number of columns in masked_weights
                block->seq_length,      // Number of columns in input, rows in masked_weights
                1.0f,                   // Alpha
                block->transposed,      // Input matrix (batch * embed_dim, seq_length)
                block->seq_length,      // Leading dimension of input
                block->masked_weights,  // Weight matrix (seq_length, seq_length) 
                block->seq_length,      // Leading dimension of weight matrix
                0.0f,                   // Beta
                block->token_mixed,     // Output matrix
                block->seq_length);     // Leading dimension of output
    
    // Apply SiLU activation: x * sigmoid(x)
    for (int i = 0; i < batch_size * block->embed_dim * block->seq_length; i++) {
        block->token_mix_activated[i] = silu(block->token_mixed[i]);
    }
    
    // Transpose back from [batch, embed, seq] to [batch, seq, embed]
    for (int b = 0; b < batch_size; b++) {
        for (int e = 0; e < block->embed_dim; e++) {
            for (int s = 0; s < block->seq_length; s++) {
                output[b * block->seq_length * block->embed_dim + s * block->embed_dim + e] = 
                    block->token_mix_activated[b * block->embed_dim * block->seq_length + e * block->seq_length + s];
            }
        }
    }
    
    // Add residual connection
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        output[i] += block->residual[i];
    }
    
    // Channel Mixing -------------------------------------
    
    // Save output as residual for channel mixing
    memcpy(block->residual, output, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Channel mixing - using optimized CBLAS GEMM
    // Treat batch_size * seq_length as a single batch dimension
    int combined_batch = batch_size * block->seq_length; // Number of vectors to process
    
    // y = alpha * A * x + beta * y
    // Using GEMM: output = 1.0 * batch_inputs * W^T + 0.0 * output
    // Where batch_inputs is (combined_batch x embed_dim) and W is (embed_dim x embed_dim)
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasNoTrans,          // No transpose for input matrix
                CblasTrans,            // Transpose weight matrix (for compatibility with row-major layout)
                combined_batch,        // Number of rows in input matrix (combined batch)
                block->embed_dim,      // Number of columns in weight matrix
                block->embed_dim,      // Number of columns in input matrix
                1.0f,                  // Alpha
                output,                // Input matrix (combined batch)
                block->embed_dim,      // Leading dimension of input matrix
                block->channel_mixing_weight, // Weight matrix
                block->embed_dim,      // Leading dimension of weight matrix
                0.0f,                  // Beta
                block->channel_mixed,  // Output matrix
                block->embed_dim);     // Leading dimension of output matrix
    
    // Apply SiLU activation
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        block->channel_mix_activated[i] = silu(block->channel_mixed[i]);
    }
    
    // Copy activated output
    memcpy(output, block->channel_mix_activated, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Add residual connection
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        output[i] += block->residual[i];
    }
}

// Forward pass through the entire model
void mixer_model_forward(MixerModel* model, int* input_tokens) {
    // Apply embedding
    apply_embedding(model, input_tokens);
    
    // Forward through each mixer block
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = &model->block_outputs[i * model->batch_size * model->seq_length * model->embed_dim];
        float* output_ptr = &model->block_outputs[(i + 1) * model->batch_size * model->seq_length * model->embed_dim];
        
        mixer_block_forward(model->blocks[i], input_ptr, output_ptr, model->batch_size);
    }
    
    // Apply output projection using CBLAS GEMM
    float* final_output = &model->block_outputs[model->num_layers * model->batch_size * model->seq_length * model->embed_dim];
    int combined_batch_seq = model->batch_size * model->seq_length;
    
    // Using GEMM: logits = final_output * out_proj_weight^T
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasNoTrans,          // No transpose for input
                CblasTrans,            // Transpose weight matrix
                combined_batch_seq,    // Number of rows in final_output (batch * seq_length)
                model->vocab_size,     // Number of columns in out_proj_weight
                model->embed_dim,      // Number of columns in final_output, rows in out_proj_weight
                1.0f,                  // Alpha
                final_output,          // Input matrix
                model->embed_dim,      // Leading dimension of input
                model->out_proj_weight,// Weight matrix
                model->embed_dim,      // Leading dimension of weight matrix
                0.0f,                  // Beta
                model->logits,         // Output matrix
                model->vocab_size);    // Leading dimension of output
}

// Compute cross-entropy loss and gradients
float compute_loss_and_gradients(MixerModel* model, int* target_tokens) {
    float total_loss = 0.0f;
    
    // Initialize d_logits with zeros
    memset(model->d_logits, 0, model->batch_size * model->seq_length * model->vocab_size * sizeof(float));
    
    // For each position, compute cross-entropy loss and gradient
    for (int b = 0; b < model->batch_size; b++) {
        for (int s = 0; s < model->seq_length; s++) {
            int target = target_tokens[b * model->seq_length + s];
            float* logits = &model->logits[b * model->seq_length * model->vocab_size + s * model->vocab_size];
            float* d_logits = &model->d_logits[b * model->seq_length * model->vocab_size + s * model->vocab_size];
            
            // Find max logit for numerical stability
            float max_logit = logits[0];
            for (int v = 1; v < model->vocab_size; v++) {
                if (logits[v] > max_logit) {
                    max_logit = logits[v];
                }
            }
            
            // Compute softmax and loss
            float sum_exp = 0.0f;
            
            for (int v = 0; v < model->vocab_size; v++) {
                model->softmax_probs[v] = expf(logits[v] - max_logit);
                sum_exp += model->softmax_probs[v];
            }
            
            for (int v = 0; v < model->vocab_size; v++) {
                model->softmax_probs[v] /= sum_exp;
            }
            
            // Cross-entropy loss for this position
            float position_loss = -logf(model->softmax_probs[target] + 1e-10f);
            total_loss += position_loss;
            
            // Gradient of loss w.r.t. logits is (softmax_prob - one_hot_target)
            for (int v = 0; v < model->vocab_size; v++) {
                d_logits[v] = model->softmax_probs[v];
            }
            d_logits[target] -= 1.0f;
        }
    }
    
    return total_loss / (model->batch_size * model->seq_length);
}

// Backward pass through the output projection layer
void output_projection_backward(MixerModel* model) {
    float* final_output = &model->block_outputs[model->num_layers * model->batch_size * model->seq_length * model->embed_dim];
    float* d_final_output = &model->d_block_outputs[model->num_layers * model->batch_size * model->seq_length * model->embed_dim];
    
    // Initialize d_final_output with zeros
    memset(d_final_output, 0, model->batch_size * model->seq_length * model->embed_dim * sizeof(float));
    
    // Zero out the output projection gradients
    memset(model->out_proj_weight_grad, 0, model->vocab_size * model->embed_dim * sizeof(float));
    
    int combined_batch_seq = model->batch_size * model->seq_length;
    
    // Gradient w.r.t. out_proj_weight
    // Using GEMM: out_proj_weight_grad = d_logits^T * final_output
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasTrans,            // Transpose d_logits
                CblasNoTrans,          // No transpose for final_output
                model->vocab_size,     // Number of rows in d_logits^T
                model->embed_dim,      // Number of columns in final_output
                combined_batch_seq,    // Number of columns in d_logits^T (= rows in final_output)
                1.0f,                  // Alpha
                model->d_logits,       // d_logits matrix
                model->vocab_size,     // Leading dimension of d_logits
                final_output,          // final_output matrix
                model->embed_dim,      // Leading dimension of final_output
                0.0f,                  // Beta
                model->out_proj_weight_grad, // Weight gradient matrix
                model->embed_dim);     // Leading dimension of weight gradient matrix
    
    // Gradient w.r.t. final_output
    // Using GEMM: d_final_output = d_logits * out_proj_weight
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasNoTrans,          // No transpose for d_logits
                CblasNoTrans,          // No transpose for weights
                combined_batch_seq,    // Number of rows in d_logits
                model->embed_dim,      // Number of columns in weights
                model->vocab_size,     // Number of columns in d_logits (= rows in weights)
                1.0f,                  // Alpha
                model->d_logits,       // d_logits matrix
                model->vocab_size,     // Leading dimension of d_logits
                model->out_proj_weight,// Weight matrix
                model->embed_dim,      // Leading dimension of weight matrix
                0.0f,                  // Beta
                d_final_output,        // Output gradient
                model->embed_dim);     // Leading dimension of output gradient
}

// Backward pass through a MixerBlock with optimized CBLAS for both token and channel mixing
void mixer_block_backward(MixerBlock* block, float* d_output, float* d_input, int batch_size) {
    // Initialize gradients for this block
    memset(block->token_mixing_weight_grad, 0, block->seq_length * block->seq_length * sizeof(float));
    memset(block->channel_mixing_weight_grad, 0, block->embed_dim * block->embed_dim * sizeof(float));
    
    // Channel Mixing Backward Pass ---------------------------------
    
    // d_output is the gradient coming from the next layer
    memcpy(block->d_output, d_output, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Gradient through the residual connection
    memcpy(block->d_channel_activated, block->d_output, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Gradient through the SiLU activation: d_out * SiLU'(x)
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        block->d_channel_mixed[i] = block->d_channel_activated[i] * silu_derivative(block->channel_mixed[i]);
    }
    
    // Channel mixing backward - optimized using CBLAS GEMM
    int combined_batch = batch_size * block->seq_length;
    
    // Gradient w.r.t. weights: W_grad = inputs^T * gradients
    // Using GEMM: W_grad = alpha * inputs^T * gradients + beta * W_grad
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasTrans,            // Transpose inputs
                CblasNoTrans,          // No transpose for gradients
                block->embed_dim,      // Number of rows in input^T matrix
                block->embed_dim,      // Number of columns in gradient matrix
                combined_batch,        // Number of columns in input^T matrix (= number of rows in gradient matrix)
                1.0f,                  // Alpha
                block->residual,       // Input matrix (saved during forward pass)
                block->embed_dim,      // Leading dimension of input matrix
                block->d_channel_mixed, // Gradient matrix
                block->embed_dim,      // Leading dimension of gradient matrix
                1.0f,                  // Beta (add to existing gradients)
                block->channel_mixing_weight_grad, // Weight gradient matrix
                block->embed_dim);     // Leading dimension of weight gradient matrix
    
    // Gradient w.r.t. inputs: d_input = gradients * W
    // Using GEMM: d_input = alpha * gradients * W + beta * d_input
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasNoTrans,          // No transpose for gradients
                CblasNoTrans,          // No transpose for weights
                combined_batch,        // Number of rows in gradient matrix
                block->embed_dim,      // Number of columns in weight matrix
                block->embed_dim,      // Number of columns in gradient matrix (= number of rows in weight matrix)
                1.0f,                  // Alpha
                block->d_channel_mixed, // Gradient matrix
                block->embed_dim,      // Leading dimension of gradient matrix
                block->channel_mixing_weight, // Weight matrix
                block->embed_dim,      // Leading dimension of weight matrix
                0.0f,                  // Beta (overwrite d_input)
                d_input,               // Input gradient matrix
                block->embed_dim);     // Leading dimension of input gradient matrix
    
    // Add the residual gradient
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        d_input[i] += block->d_output[i];
    }
    
    // Token Mixing Backward Pass -----------------------------------
    
    // Gradient coming from the channel mixing backward pass is in d_input
    memcpy(block->d_output, d_input, batch_size * block->seq_length * block->embed_dim * sizeof(float));
    
    // Transpose d_output for token mixing backwards: [batch, seq, embed] -> [batch, embed, seq]
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < block->seq_length; s++) {
            for (int e = 0; e < block->embed_dim; e++) {
                block->d_output_transposed[b * block->embed_dim * block->seq_length + e * block->seq_length + s] = 
                    block->d_output[b * block->seq_length * block->embed_dim + s * block->embed_dim + e];
            }
        }
    }
    
    // Gradient through SiLU activation for token mixing
    for (int i = 0; i < batch_size * block->embed_dim * block->seq_length; i++) {
        block->d_token_mixed[i] = block->d_output_transposed[i] * silu_derivative(block->token_mixed[i]);
    }
    
    // Combined batch * embed_dim dimension
    int combined_batch_embed = batch_size * block->embed_dim;
    
    // Gradient w.r.t. token mixing weights: W_grad = d_token_mixed^T * transposed
    // Using GEMM: W_grad = alpha * d_token_mixed^T * transposed + beta * W_grad
    memset(block->temp_grad, 0, block->seq_length * block->seq_length * sizeof(float));
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasTrans,            // Transpose d_token_mixed
                CblasNoTrans,          // No transpose for transposed input
                block->seq_length,     // Number of rows in d_token_mixed^T
                block->seq_length,     // Number of columns in transposed
                combined_batch_embed,  // Number of columns in d_token_mixed^T (= rows in transposed)
                1.0f,                  // Alpha
                block->d_token_mixed,  // d_token_mixed matrix
                block->seq_length,     // Leading dimension of d_token_mixed
                block->transposed,     // Transposed inputs from forward pass
                block->seq_length,     // Leading dimension of transposed input
                0.0f,                  // Beta
                block->temp_grad,      // Temporary weight gradient matrix
                block->seq_length);    // Leading dimension of temp_grad

    // Apply mask to the gradients and add to weight_grad
    for (int i = 0; i < block->seq_length; i++) {
        for (int j = 0; j < block->seq_length; j++) {
            if (block->token_mixing_mask[i * block->seq_length + j] > 0) {
                block->token_mixing_weight_grad[i * block->seq_length + j] += block->temp_grad[i * block->seq_length + j];
            }
        }
    }
    
    // Gradient w.r.t. input (transposed): d_input_transposed = d_token_mixed * masked_weights
    // Using GEMM: d_input_transposed = alpha * d_token_mixed * masked_weights + beta * d_input_transposed
    cblas_sgemm(CblasRowMajor,         // Layout (row-major)
                CblasNoTrans,          // No transpose for d_token_mixed
                CblasNoTrans,          // No transpose for masked_weights
                combined_batch_embed,  // Number of rows in d_token_mixed
                block->seq_length,     // Number of columns in masked_weights
                block->seq_length,     // Number of columns in d_token_mixed (= rows in masked_weights)
                1.0f,                  // Alpha
                block->d_token_mixed,  // d_token_mixed matrix
                block->seq_length,     // Leading dimension of d_token_mixed
                block->masked_weights, // Masked weight matrix
                block->seq_length,     // Leading dimension of masked weights
                0.0f,                  // Beta
                block->d_input_transposed, // d_input_transposed matrix
                block->seq_length);    // Leading dimension of d_input_transposed
                
    // Transpose the gradient back to the original format [batch, embed, seq] -> [batch, seq, embed]
    for (int b = 0; b < batch_size; b++) {
        for (int e = 0; e < block->embed_dim; e++) {
            for (int s = 0; s < block->seq_length; s++) {
                d_input[b * block->seq_length * block->embed_dim + s * block->embed_dim + e] = 
                    block->d_input_transposed[b * block->embed_dim * block->seq_length + e * block->seq_length + s];
            }
        }
    }
    
    // Add the residual gradient from token mixing
    for (int i = 0; i < batch_size * block->seq_length * block->embed_dim; i++) {
        d_input[i] += block->d_output[i];
    }
}

// Backward pass through the embedding layer
void embedding_backward(MixerModel* model, int* input_tokens) {
    // We've computed d_embedding in the first mixer block's backward pass
    float* d_embeddings = &model->d_block_outputs[0];
    
    // Zero out embedding gradients
    memset(model->embedding_weight_grad, 0, model->vocab_size * model->embed_dim * sizeof(float));
    
    // Accumulate gradients for each token in the batch
    for (int b = 0; b < model->batch_size; b++) {
        for (int s = 0; s < model->seq_length; s++) {
            int token_idx = input_tokens[b * model->seq_length + s];
            float* d_embed_pos = &d_embeddings[b * model->seq_length * model->embed_dim + s * model->embed_dim];
            
            for (int e = 0; e < model->embed_dim; e++) {
                model->embedding_weight_grad[token_idx * model->embed_dim + e] += d_embed_pos[e];
            }
        }
    }
}

// Backward pass through the entire model
void mixer_model_backward(MixerModel* model, int* input_tokens, int* target_tokens) {
    // Compute loss and gradients for output logits
    compute_loss_and_gradients(model, target_tokens);
    
    // Backward pass through output projection
    output_projection_backward(model);
    
    // Backward pass through mixer blocks
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output = &model->d_block_outputs[(i + 1) * model->batch_size * model->seq_length * model->embed_dim];
        float* d_input = &model->d_block_outputs[i * model->batch_size * model->seq_length * model->embed_dim];
        
        mixer_block_backward(model->blocks[i], d_output, d_input, model->batch_size);
    }
    
    // Backward pass through embedding layer
    embedding_backward(model, input_tokens);
}

// Update weights using AdamW optimizer
void update_weights_adamw(MixerModel* model, float learning_rate) {
    model->t++;  // Increment time step
    
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update embedding weights
    for (int i = 0; i < model->vocab_size * model->embed_dim; i++) {
        float grad = model->embedding_weight_grad[i] / (model->batch_size * model->seq_length);
        
        model->embedding_m[i] = model->beta1 * model->embedding_m[i] + (1.0f - model->beta1) * grad;
        model->embedding_v[i] = model->beta2 * model->embedding_v[i] + (1.0f - model->beta2) * grad * grad;
        
        float update = alpha_t * model->embedding_m[i] / (sqrtf(model->embedding_v[i]) + model->epsilon);
        model->embedding_weight[i] = model->embedding_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
    }
    
    // Update output projection weights
    for (int i = 0; i < model->vocab_size * model->embed_dim; i++) {
        float grad = model->out_proj_weight_grad[i] / (model->batch_size * model->seq_length);
        
        model->out_proj_m[i] = model->beta1 * model->out_proj_m[i] + (1.0f - model->beta1) * grad;
        model->out_proj_v[i] = model->beta2 * model->out_proj_v[i] + (1.0f - model->beta2) * grad * grad;
        
        float update = alpha_t * model->out_proj_m[i] / (sqrtf(model->out_proj_v[i]) + model->epsilon);
        model->out_proj_weight[i] = model->out_proj_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
    }
    
    // Update mixer block weights
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        
        // Update token mixing weights
        for (int i = 0; i < block->seq_length * block->seq_length; i++) {
            // Only update if this weight is used (not masked out)
            if (block->token_mixing_mask[i] > 0) {
                float grad = block->token_mixing_weight_grad[i] / (model->batch_size * model->seq_length);
                
                block->token_mixing_m[i] = model->beta1 * block->token_mixing_m[i] + (1.0f - model->beta1) * grad;
                block->token_mixing_v[i] = model->beta2 * block->token_mixing_v[i] + (1.0f - model->beta2) * grad * grad;
                
                float update = alpha_t * block->token_mixing_m[i] / (sqrtf(block->token_mixing_v[i]) + model->epsilon);
                block->token_mixing_weight[i] = block->token_mixing_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
            }
        }
        
        // Update channel mixing weights
        for (int i = 0; i < block->embed_dim * block->embed_dim; i++) {
            float grad = block->channel_mixing_weight_grad[i] / (model->batch_size * model->seq_length);
            
            block->channel_mixing_m[i] = model->beta1 * block->channel_mixing_m[i] + (1.0f - model->beta1) * grad;
            block->channel_mixing_v[i] = model->beta2 * block->channel_mixing_v[i] + (1.0f - model->beta2) * grad * grad;
            
            float update = alpha_t * block->channel_mixing_m[i] / (sqrtf(block->channel_mixing_v[i]) + model->epsilon);
            block->channel_mixing_weight[i] = block->channel_mixing_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
        }
    }
}

// Zero out all gradients
void zero_gradients(MixerModel* model) {
    memset(model->embedding_weight_grad, 0, model->vocab_size * model->embed_dim * sizeof(float));
    memset(model->out_proj_weight_grad, 0, model->vocab_size * model->embed_dim * sizeof(float));
    
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        memset(block->token_mixing_weight_grad, 0, block->seq_length * block->seq_length * sizeof(float));
        memset(block->channel_mixing_weight_grad, 0, block->embed_dim * block->embed_dim * sizeof(float));
    }
}

// Function to apply softmax on a vector
void softmax(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf((input[i] - max_val) / TEMPERATURE);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Function to sample from a distribution
int sample_from_distribution(float* probs, int size) {
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    
    for (int i = 0; i < size; i++) {
        cdf += probs[i];
        if (r < cdf) {
            return i;
        }
    }
    
    return size - 1;  // Fallback to the last token
}

// Function to encode a UTF-8 string to byte tokens
int encode_string(const char* input, int* tokens, int max_tokens) {
    int len = 0;
    while (*input && len < max_tokens) {
        tokens[len++] = (unsigned char)(*input++);
    }
    return len;
}

// Function to decode byte tokens to a UTF-8 string
void decode_tokens(int* tokens, int num_tokens, char* output, int max_length) {
    int len = 0;
    for (int i = 0; i < num_tokens && len < max_length - 1; i++) {
        output[len++] = (char)(tokens[i] & 0xFF);
    }
    output[len] = '\0';
}

// Function to generate text from the model
void generate_text(MixerModel* model, const char* seed_text, int max_length, char* output, int output_size) {
    int* seed_tokens = (int*)malloc(model->seq_length * sizeof(int));
    int* input_batch = (int*)malloc(model->batch_size * model->seq_length * sizeof(int));
    float* probs = (float*)malloc(model->vocab_size * sizeof(float));
    
    // Zero-initialize tokens
    memset(seed_tokens, 0, model->seq_length * sizeof(int));
    
    // Encode the seed text
    int seed_length = encode_string(seed_text, seed_tokens, model->seq_length);
    
    // If seed is shorter than seq_length, shift it to the end
    if (seed_length < model->seq_length) {
        memmove(seed_tokens + (model->seq_length - seed_length), seed_tokens, seed_length * sizeof(int));
        memset(seed_tokens, 0, (model->seq_length - seed_length) * sizeof(int));
    }
    
    // Copy seed tokens to the first row of the input batch
    for (int i = 0; i < model->seq_length; i++) {
        input_batch[i] = seed_tokens[i];
    }
    
    // Zero-initialize the rest of the batch
    for (int b = 1; b < model->batch_size; b++) {
        for (int s = 0; s < model->seq_length; s++) {
            input_batch[b * model->seq_length + s] = 0;
        }
    }
    
    // Allocate buffer for generated tokens
    int* generated_tokens = (int*)malloc(max_length * sizeof(int));
    
    // Generate text token by token
    for (int i = 0; i < max_length; i++) {
        // Forward pass
        mixer_model_forward(model, input_batch);
        
        // Get logits for the last token position of the first example
        float* last_position_logits = &model->logits[0 * model->seq_length * model->vocab_size + 
                                                   (model->seq_length - 1) * model->vocab_size];
        
        // Apply softmax to get probabilities
        softmax(last_position_logits, probs, model->vocab_size);
        
        // Sample the next token
        int next_token = sample_from_distribution(probs, model->vocab_size);
        generated_tokens[i] = next_token;
        
        // Shift input tokens to the left and add the new token at the end
        for (int s = 0; s < model->seq_length - 1; s++) {
            input_batch[s] = input_batch[s + 1];
        }
        input_batch[model->seq_length - 1] = next_token;
    }
    
    // Decode the generated tokens
    decode_tokens(generated_tokens, max_length, output, output_size);
    
    free(seed_tokens);
    free(input_batch);
    free(probs);
    free(generated_tokens);
}

// Count model parameters
int count_parameters(MixerModel* model) {
    int total_params = 0;
    
    // Embedding weights
    total_params += model->vocab_size * model->embed_dim;
    
    // Output projection weights
    total_params += model->vocab_size * model->embed_dim;
    
    // MixerBlock parameters
    for (int i = 0; i < model->num_layers; i++) {
        // Token mixing weights
        total_params += model->seq_length * model->seq_length;
        
        // Channel mixing weights
        total_params += model->embed_dim * model->embed_dim;
    }
    
    return total_params;
}

// Load text data from a file
char* load_text_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error: could not read entire file\n");
        free(buffer);
        fclose(file);
        return NULL;
    }
    
    buffer[file_size] = '\0';
    *size = file_size;
    
    fclose(file);
    return buffer;
}

// Create a batch of samples for training
void create_batch(char* text, size_t text_size, int seq_length, int batch_size, int* input_tokens, int* target_tokens) {
    for (int b = 0; b < batch_size; b++) {
        // Choose a random starting position in the text
        size_t start_pos = rand() % (text_size - seq_length - 1);
        
        // Fill in input tokens (X)
        for (int s = 0; s < seq_length; s++) {
            input_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s];
        }
        
        // Fill in target tokens (Y) - shifted by 1
        for (int s = 0; s < seq_length; s++) {
            target_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s + 1];
        }
    }
}

// Save model to a binary file
void save_model(MixerModel* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        return;
    }
    
    // Write model dimensions
    fwrite(&model->vocab_size, sizeof(int), 1, file);
    fwrite(&model->embed_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    fwrite(&model->batch_size, sizeof(int), 1, file);
    
    // Write embedding weights
    fwrite(model->embedding_weight, sizeof(float), model->vocab_size * model->embed_dim, file);
    
    // Write output projection weights
    fwrite(model->out_proj_weight, sizeof(float), model->vocab_size * model->embed_dim, file);
    
    // Write MixerBlock parameters
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        
        // Write token mixing weights
        fwrite(block->token_mixing_weight, sizeof(float), block->seq_length * block->seq_length, file);
        
        // Write channel mixing weights
        fwrite(block->channel_mixing_weight, sizeof(float), block->embed_dim * block->embed_dim, file);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model from a binary file
MixerModel* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for reading\n", filename);
        return NULL;
    }
    
    // Read model dimensions
    int vocab_size, embed_dim, num_layers, seq_length, batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize model with loaded dimensions
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    
    // Read embedding weights
    fread(model->embedding_weight, sizeof(float), model->vocab_size * model->embed_dim, file);
    
    // Read output projection weights
    fread(model->out_proj_weight, sizeof(float), model->vocab_size * model->embed_dim, file);
    
    // Read MixerBlock parameters
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        
        // Read token mixing weights
        fread(block->token_mixing_weight, sizeof(float), block->seq_length * block->seq_length, file);
        
        // Read channel mixing weights
        fread(block->channel_mixing_weight, sizeof(float), block->embed_dim * block->embed_dim, file);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return model;
}

// Main function
int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);
    
    printf("Initializing Mixer model...\n");
    
    // Model hyperparameters
    int vocab_size = MAX_VOCAB_SIZE;
    int embed_dim = 128;
    int num_layers = 4;
    int seq_length = 512;
    int batch_size = 16;
    
    // Initialize model
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    
    // Count and print model parameters
    int num_params = count_parameters(model);
    printf("Model initialized with %d parameters\n", num_params);
    
    // Load text data from file
    size_t text_size;
    char* text = load_text_file("gutenberg_texts/combined_corpus.txt", &text_size);
    if (!text) {
        fprintf(stderr, "Failed to load text data\n");
        free_mixer_model(model);
        return 1;
    }
    
    printf("Loaded text corpus with %zu bytes\n", text_size);

    // Training parameters
    float learning_rate = 0.0001f;
    int num_epochs = 10;
    int steps_per_epoch = text_size / (seq_length * batch_size);
    
    // Allocate memory for batches
    int* input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    // Extract a sample for text generation
    char seed_text[1024];
    size_t seed_pos = rand() % (text_size - 1024);
    strncpy(seed_text, text + seed_pos, 1023);
    seed_text[1023] = '\0';
    char generated_text[2048];

    // Training loop
    printf("Starting training...\n");
    time_t start_time = time(NULL);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int step = 0; step < steps_per_epoch; step++) {
            // Create a new batch
            create_batch(text, text_size, seq_length, batch_size, input_tokens, target_tokens);
            
            // Forward pass
            mixer_model_forward(model, input_tokens);
            
            // Backward pass
            mixer_model_backward(model, input_tokens, target_tokens);
            
            // Update weights
            update_weights_adamw(model, learning_rate);
            
            // Calculate loss for reporting
            float step_loss = compute_loss_and_gradients(model, target_tokens);
            epoch_loss += step_loss;
            
            if (step % 10 == 0) {
                printf("Epoch %d/%d, Step %d/%d, Loss: %.4f\n", 
                       epoch+1, num_epochs, step+1, steps_per_epoch, step_loss);
            }

            if(step % 100 == 0) {
                printf("Generating sample text periodically...\n");
                printf("Sample seed text for generation:\n%s\n\n", seed_text);
                generate_text(model, seed_text, 100, generated_text, sizeof(generated_text));
                printf("Generated text:\n%s\n\n", generated_text);
            }
        }
        
        epoch_loss /= steps_per_epoch;
        time_t current_time = time(NULL);
        
        printf("\nEpoch %d/%d completed, Average Loss: %.4f, Time elapsed: %ld seconds\n\n", 
               epoch+1, num_epochs, epoch_loss, current_time - start_time);
    }
    
    // Final model saving
    save_model(model, "mixer_model_final.bin");
    
    // Clean up
    free(input_tokens);
    free(target_tokens);
    free(text);
    free_mixer_model(model);
    
    printf("Training completed!\n");
    
    return 0;
}