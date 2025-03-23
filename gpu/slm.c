#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// MixerBlock structure – all pointers refer to device memory.
typedef struct {
    // Token mixing parameters
    float* token_mixing_weight;         // [seq_length x seq_length]
    float* token_mixing_weight_grad;
    float* token_mixing_mask;           // [seq_length x seq_length]
    float* token_mixing_m;
    float* token_mixing_v;
    
    // Channel mixing parameters
    float* channel_mixing_weight;       // [embed_dim x embed_dim]
    float* channel_mixing_weight_grad;
    float* channel_mixing_m;
    float* channel_mixing_v;
    
    // Forward pass buffers
    float* input_buffer;   // [batch, seq, embed]
    float* residual;       // [batch, seq, embed]
    float* transposed;     // [batch, embed, seq] temporary for token mixing
    float* token_mixed;    // [batch, embed, seq]
    float* token_mix_activated; // [batch, embed, seq]
    float* channel_mixed;  // [batch, seq, embed]
    float* channel_mix_activated; // [batch, seq, embed]
    
    // Backward pass buffers
    float* d_output;       // gradient from next layer [batch, seq, embed]
    float* d_token_mixed;  // [batch, embed, seq]
    float* d_channel_mixed; // [batch, seq, embed]
    float* d_input;         // [batch, seq, embed]
    
    // Additional buffers
    float* masked_weights;   // [seq, seq] = token_mixing_weight * token_mixing_mask
    float* d_channel_activated; // [batch, seq, embed]
    float* d_output_transposed; // [batch, embed, seq]
    float* d_input_transposed;  // [batch, embed, seq]
    float* temp_grad;           // [seq, seq] temporary
     
    // Dimensions:
    int embed_dim;
    int seq_length;
} MixerBlock;

// MixerModel structure.
typedef struct {
    // Embedding parameters
    float* embedding_weight;          // [vocab_size x embed_dim]
    float* embedding_weight_grad;
    float* embedding_m;
    float* embedding_v;
    
    // Output projection parameters
    float* out_proj_weight;           // [vocab_size x embed_dim]
    float* out_proj_weight_grad;
    float* out_proj_m;
    float* out_proj_v;
    
    // Array of MixerBlocks
    MixerBlock** blocks;
    
    // Forward pass buffers
    float* embeddings;     // [batch, seq, embed]
    float* block_outputs;  // (num_layers+1) concatenated buffers, each [batch, seq, embed]
    float* logits;         // [batch, seq, vocab_size]
    
    // Backward pass buffers
    float* d_logits;       // [batch, seq, vocab_size]
    float* d_block_outputs;// same layout as block_outputs
    
    // Persistent device buffers for training
    int* d_input_tokens;
    int* d_target_tokens;
    float* d_loss_buffer;
    
    // Persistent CPU buffers for training
    float* h_loss_buffer;
    float* h_out_proj_weight;
    float* h_embedding_weight;
    
    // Additional preallocated buffer for softmax (on host)
    float* softmax_probs;  // [vocab_size]
    
    // Adam optimizer parameters
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;  // time step counter
    
    // Dimensions/hyperparameters
    int vocab_size;
    int embed_dim;
    int num_layers;
    int seq_length;
    int batch_size;
} MixerModel;

// Dataset structure to store text and manage pre-shuffled indices.
typedef struct {
    char* text;               // Raw text data (as bytes)
    size_t text_size;         // Size of the text in bytes
    int seq_length;           // Sequence length (in tokens)
    int* indices;             // Array of valid starting indices (in token space)
    int num_indices;          // Number of valid indices
    int* shuffled_indices;    // Shuffled array of indices for current epoch
    int current_position;     // Current position in the shuffled indices
} Dataset;

////////////////////////////////////////////////////////////////////////////////
// Device helper functions and CUDA kernels

// Kernel: elementwise SiLU activation: out[i] = x * sigmoid(x)
__global__ void kernel_silu(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

// Kernel: multiply gradient by SiLU derivative: grad_out = grad_in * silu_deriv(pre[i])
__global__ void kernel_silu_deriv_mult(const float* pre, const float* grad_in, float* grad_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = pre[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        float deriv = sig + x * sig * (1.0f - sig);
        grad_out[idx] = grad_in[idx] * deriv;
    }
}

// Kernel: transpose from [batch, seq, embed] to [batch, embed, seq]
__global__ void kernel_transpose_BSE_to_BES(const float* input, float* output, int batch, int seq, int embed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * embed;
    if(idx < total) {
        int b = idx / (seq * embed);
        int rem = idx % (seq * embed);
        int s = rem / embed;
        int e = rem % embed;
        output[b * embed * seq + e * seq + s] = input[b * seq * embed + s * embed + e];
    }
}

// Kernel: transpose from [batch, embed, seq] to [batch, seq, embed]
__global__ void kernel_transpose_BES_to_BSE(const float* input, float* output, int batch, int embed, int seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * embed * seq;
    if(idx < total) {
        int b = idx / (embed * seq);
        int rem = idx % (embed * seq);
        int e = rem / seq;
        int s = rem % seq;
        output[b * seq * embed + s * embed + e] = input[b * embed * seq + e * seq + s];
    }
}

// Kernel: elementwise multiply two arrays (for masking)
__global__ void kernel_apply_mask(const float* weights, const float* mask, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = weights[idx] * mask[idx];
    }
}

// Kernel: embedding lookup. For each token in input, copy its embedding row into output.
__global__ void kernel_apply_embedding(const int* input_tokens, const float* embedding_weight, float* embeddings,
                                         int batch_size, int seq_length, int embed_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_length;
    if(idx < total) {
        int token = input_tokens[idx];
        if(token < 0 || token >= vocab_size) token = 0;
        for (int e = 0; e < embed_dim; e++) {
            embeddings[idx * embed_dim + e] = embedding_weight[token * embed_dim + e];
        }
    }
}

// Kernel for Adam weight update
__global__ void kernel_adam_update(float* weight, const float* grad, float* m, float* v,
                                  float alpha_t, float beta1, float beta2, float epsilon,
                                  float weight_decay, float lr, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float g = grad[idx] / scale;
        float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
        float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = m_new;
        v[idx] = v_new;
        float update = alpha_t * m_new / (sqrtf(v_new) + epsilon);
        weight[idx] = weight[idx] * (1.0f - lr * weight_decay) - update;
    }
}

// Kernel to compute softmax and cross-entropy loss
__global__ void kernel_softmax_cross_entropy(const float* logits, const int* targets,
                                           float* d_logits, float* loss_buffer,
                                           int batch_size, int seq_length, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length) {
        int target = targets[idx];
        
        // Find max for numerical stability
        float max_val = logits[idx * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[idx * vocab_size + v];
            max_val = (val > max_val) ? val : max_val;
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float exp_val = expf(logits[idx * vocab_size + v] - max_val);
            d_logits[idx * vocab_size + v] = exp_val;  // Temporarily store exp values
            sum_exp += exp_val;
        }
        
        // Normalize and compute gradient and loss
        float loss = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float prob = d_logits[idx * vocab_size + v] / sum_exp;
            d_logits[idx * vocab_size + v] = prob;  // Store normalized probability
            
            // If this is the target token, subtract 1 for gradient and compute log loss
            if (v == target) {
                d_logits[idx * vocab_size + v] -= 1.0f;
                loss = -logf(prob + 1e-10f);
            }
        }
        
        // Store loss for this sample
        loss_buffer[idx] = loss;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Initialization and free functions for MixerBlock and MixerModel.

MixerBlock* init_mixer_block(int embed_dim, int seq_length, int batch_size) {
    MixerBlock* block = (MixerBlock*)malloc(sizeof(MixerBlock));
    block->embed_dim = embed_dim;
    block->seq_length = seq_length;
    
    size_t size_tok = seq_length * seq_length * sizeof(float);
    // Initialize token mixing weights on host then copy to device.
    float* h_token_mixing_weight = (float*)malloc(size_tok);
    float scale_token = 1.0f / sqrtf((float)seq_length);
    for (int i = 0; i < seq_length * seq_length; i++) {
        h_token_mixing_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_token);
    }
    CHECK_CUDA(cudaMalloc(&block->token_mixing_weight, size_tok));
    CHECK_CUDA(cudaMemcpy(block->token_mixing_weight, h_token_mixing_weight, size_tok, cudaMemcpyHostToDevice));
    free(h_token_mixing_weight);
    
    CHECK_CUDA(cudaMalloc(&block->token_mixing_weight_grad, size_tok));
    CHECK_CUDA(cudaMemset(block->token_mixing_weight_grad, 0, size_tok));
    CHECK_CUDA(cudaMalloc(&block->token_mixing_m, size_tok));
    CHECK_CUDA(cudaMemset(block->token_mixing_m, 0, size_tok));
    CHECK_CUDA(cudaMalloc(&block->token_mixing_v, size_tok));
    CHECK_CUDA(cudaMemset(block->token_mixing_v, 0, size_tok));
    
    // Create token mixing mask on host then copy.
    float* h_token_mixing_mask = (float*)malloc(size_tok);
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            h_token_mixing_mask[i * seq_length + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
    CHECK_CUDA(cudaMalloc(&block->token_mixing_mask, size_tok));
    CHECK_CUDA(cudaMemcpy(block->token_mixing_mask, h_token_mixing_mask, size_tok, cudaMemcpyHostToDevice));
    free(h_token_mixing_mask);
    
    // Pre-allocate masked_weights buffer
    CHECK_CUDA(cudaMalloc(&block->masked_weights, size_tok));
    
    // Channel mixing weights.
    size_t size_channel = embed_dim * embed_dim * sizeof(float);
    float* h_channel_mixing_weight = (float*)malloc(size_channel);
    float scale_channel = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < embed_dim * embed_dim; i++){
        h_channel_mixing_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_channel);
    }
    CHECK_CUDA(cudaMalloc(&block->channel_mixing_weight, size_channel));
    CHECK_CUDA(cudaMemcpy(block->channel_mixing_weight, h_channel_mixing_weight, size_channel, cudaMemcpyHostToDevice));
    free(h_channel_mixing_weight);
    
    CHECK_CUDA(cudaMalloc(&block->channel_mixing_weight_grad, size_channel));
    CHECK_CUDA(cudaMemset(block->channel_mixing_weight_grad, 0, size_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_mixing_m, size_channel));
    CHECK_CUDA(cudaMemset(block->channel_mixing_m, 0, size_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_mixing_v, size_channel));
    CHECK_CUDA(cudaMemset(block->channel_mixing_v, 0, size_channel));
    
    // Allocate forward pass buffers.
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&block->input_buffer, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->residual, tensor_size));
    size_t tensor_size_trans = batch_size * embed_dim * seq_length * sizeof(float);
    CHECK_CUDA(cudaMalloc(&block->transposed, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->token_mixed, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->token_mix_activated, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->channel_mixed, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->channel_mix_activated, tensor_size));
    
    // Allocate backward pass buffers.
    CHECK_CUDA(cudaMalloc(&block->d_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_token_mixed, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->d_channel_mixed, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_input, tensor_size));
    
    // Additional buffers.
    CHECK_CUDA(cudaMalloc(&block->d_channel_activated, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_output_transposed, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->d_input_transposed, tensor_size_trans));
    CHECK_CUDA(cudaMalloc(&block->temp_grad, size_tok));
    CHECK_CUDA(cudaMemset(block->temp_grad, 0, size_tok));
    
    return block;
}

void free_mixer_block(MixerBlock* block) {
    CHECK_CUDA(cudaFree(block->token_mixing_weight));
    CHECK_CUDA(cudaFree(block->token_mixing_weight_grad));
    CHECK_CUDA(cudaFree(block->token_mixing_mask));
    CHECK_CUDA(cudaFree(block->token_mixing_m));
    CHECK_CUDA(cudaFree(block->token_mixing_v));
    
    CHECK_CUDA(cudaFree(block->channel_mixing_weight));
    CHECK_CUDA(cudaFree(block->channel_mixing_weight_grad));
    CHECK_CUDA(cudaFree(block->channel_mixing_m));
    CHECK_CUDA(cudaFree(block->channel_mixing_v));
    
    CHECK_CUDA(cudaFree(block->input_buffer));
    CHECK_CUDA(cudaFree(block->residual));
    CHECK_CUDA(cudaFree(block->transposed));
    CHECK_CUDA(cudaFree(block->token_mixed));
    CHECK_CUDA(cudaFree(block->token_mix_activated));
    CHECK_CUDA(cudaFree(block->channel_mixed));
    CHECK_CUDA(cudaFree(block->channel_mix_activated));
    
    CHECK_CUDA(cudaFree(block->d_output));
    CHECK_CUDA(cudaFree(block->d_token_mixed));
    CHECK_CUDA(cudaFree(block->d_channel_mixed));
    CHECK_CUDA(cudaFree(block->d_input));
    
    CHECK_CUDA(cudaFree(block->masked_weights));
    CHECK_CUDA(cudaFree(block->d_channel_activated));
    CHECK_CUDA(cudaFree(block->d_output_transposed));
    CHECK_CUDA(cudaFree(block->d_input_transposed));
    CHECK_CUDA(cudaFree(block->temp_grad));
    
    free(block);
}

MixerModel* init_mixer_model(int vocab_size, int embed_dim, int num_layers, int seq_length, int batch_size) {
    MixerModel* model = (MixerModel*)malloc(sizeof(MixerModel));
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->num_layers = num_layers;
    model->seq_length = seq_length;
    model->batch_size = batch_size;
    
    model->beta1 = 0.9f;
    model->beta2 = 0.999f;
    model->epsilon = 1e-8f;
    model->weight_decay = 0.01f;
    model->t = 0;
    
    size_t embed_matrix_size = vocab_size * embed_dim * sizeof(float);
    float* h_embedding_weight = (float*)malloc(embed_matrix_size);
    float scale_embed = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_embedding_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    }
    CHECK_CUDA(cudaMalloc(&model->embedding_weight, embed_matrix_size));
    CHECK_CUDA(cudaMemcpy(model->embedding_weight, h_embedding_weight, embed_matrix_size, cudaMemcpyHostToDevice));
    
    // Keep a CPU copy for quick parameter access
    model->h_embedding_weight = h_embedding_weight;
    
    CHECK_CUDA(cudaMalloc(&model->embedding_weight_grad, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->embedding_m, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_m, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->embedding_v, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_v, 0, embed_matrix_size));
    
    // Output projection weights.
    CHECK_CUDA(cudaMalloc(&model->out_proj_weight, embed_matrix_size));
    float* h_out_proj_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_out_proj_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    }
    CHECK_CUDA(cudaMemcpy(model->out_proj_weight, h_out_proj_weight, embed_matrix_size, cudaMemcpyHostToDevice));
    
    // Keep a CPU copy for quick parameter access
    model->h_out_proj_weight = h_out_proj_weight;
    
    CHECK_CUDA(cudaMalloc(&model->out_proj_weight_grad, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->out_proj_m, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_m, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->out_proj_v, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_v, 0, embed_matrix_size));
    
    // Allocate MixerBlocks.
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_mixer_block(embed_dim, seq_length, batch_size);
    }
    
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&model->embeddings, tensor_size));
    // Allocate (num_layers+1) outputs concatenated.
    CHECK_CUDA(cudaMalloc(&model->block_outputs, (num_layers+1) * tensor_size));
    // Logits: [batch, seq, vocab_size]
    size_t logits_size = batch_size * seq_length * vocab_size * sizeof(float);
    CHECK_CUDA(cudaMalloc(&model->logits, logits_size));
    
    CHECK_CUDA(cudaMalloc(&model->d_logits, logits_size));
    CHECK_CUDA(cudaMalloc(&model->d_block_outputs, (num_layers+1) * tensor_size));
    
    // Allocate persistent device buffers for training
    CHECK_CUDA(cudaMalloc(&model->d_input_tokens, batch_size * seq_length * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_target_tokens, batch_size * seq_length * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_loss_buffer, batch_size * seq_length * sizeof(float)));
    
    // Allocate persistent host buffers for loss computation
    model->h_loss_buffer = (float*)malloc(batch_size * seq_length * sizeof(float));
    
    // softmax_probs on CPU
    model->softmax_probs = (float*)malloc(vocab_size * sizeof(float));
    
    return model;
}

void free_mixer_model(MixerModel* model) {
    CHECK_CUDA(cudaFree(model->embedding_weight));
    CHECK_CUDA(cudaFree(model->embedding_weight_grad));
    CHECK_CUDA(cudaFree(model->embedding_m));
    CHECK_CUDA(cudaFree(model->embedding_v));
    
    CHECK_CUDA(cudaFree(model->out_proj_weight));
    CHECK_CUDA(cudaFree(model->out_proj_weight_grad));
    CHECK_CUDA(cudaFree(model->out_proj_m));
    CHECK_CUDA(cudaFree(model->out_proj_v));
    
    for (int i = 0; i < model->num_layers; i++) {
        free_mixer_block(model->blocks[i]);
    }
    free(model->blocks);
    
    CHECK_CUDA(cudaFree(model->embeddings));
    CHECK_CUDA(cudaFree(model->block_outputs));
    CHECK_CUDA(cudaFree(model->logits));
    
    CHECK_CUDA(cudaFree(model->d_logits));
    CHECK_CUDA(cudaFree(model->d_block_outputs));
    
    // Free persistent buffers
    CHECK_CUDA(cudaFree(model->d_input_tokens));
    CHECK_CUDA(cudaFree(model->d_target_tokens));
    CHECK_CUDA(cudaFree(model->d_loss_buffer));
    
    free(model->h_loss_buffer);
    free(model->h_embedding_weight);
    free(model->h_out_proj_weight);
    free(model->softmax_probs);
    
    free(model);
}

////////////////////////////////////////////////////////////////////////////////
// Forward pass through one MixerBlock.
// Performs token mixing (with transposition, GEMM using masked weights, SiLU,
// transposing the result back and adding the residual) and then channel mixing.
void mixer_block_forward(cublasHandle_t handle, MixerBlock* block, float* input, float* output, int batch_size) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int total = batch_size * seq * embed;
    int total_trans = batch_size * embed * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    // Save input for backward.
    CHECK_CUDA(cudaMemcpy(block->input_buffer, input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(block->residual, input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Transpose input: [batch, seq, embed] -> [batch, embed, seq]
    int threads = 256;
    int nblocks = (total + threads - 1) / threads;
    kernel_transpose_BSE_to_BES<<<nblocks, threads>>>(input, block->transposed, batch_size, seq, embed);
    
    // Apply masking to the token mixing weights
    int mask_blocks = (seq * seq + threads - 1) / threads;
    kernel_apply_mask<<<mask_blocks, threads>>>(block->token_mixing_weight, block->token_mixing_mask, block->masked_weights, seq * seq);
    
    // Token mixing GEMM:
    // Compute: token_mixed = transposed * (masked_weights)^T.
    int combined_batch_embed = batch_size * embed;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                seq, combined_batch_embed, seq,
                &alpha,
                block->masked_weights, seq,
                block->transposed, seq,
                &beta,
                block->token_mixed, seq));
    
    // Apply SiLU activation.
    nblocks = (total_trans + threads - 1) / threads;
    kernel_silu<<<nblocks, threads>>>(block->token_mixed, block->token_mix_activated, total_trans);
    
    // Transpose back: [batch, embed, seq] -> [batch, seq, embed]
    nblocks = (total + threads - 1) / threads;
    kernel_transpose_BES_to_BSE<<<nblocks, threads>>>(block->token_mix_activated, output, batch_size, embed, seq);
    
    // Add residual
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->residual, 1, output, 1));
    
    // --- Channel Mixing ---
    // Save current output as residual.
    CHECK_CUDA(cudaMemcpy(block->residual, output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Channel mixing GEMM: channel_mixed = output * (channel_mixing_weight)^T.
    int combined_batch = batch_size * seq;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                embed, combined_batch, embed,
                &alpha,
                block->channel_mixing_weight, embed,
                output, embed,
                &beta,
                block->channel_mixed, embed));
    
    // Apply SiLU activation.
    nblocks = (total + threads - 1) / threads;
    kernel_silu<<<nblocks, threads>>>(block->channel_mixed, block->channel_mix_activated, total);
    
    // Copy activated output
    CHECK_CUDA(cudaMemcpy(output, block->channel_mix_activated, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Add residual
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->residual, 1, output, 1));
}

//
// Forward pass through the entire model.
void mixer_model_forward(cublasHandle_t handle, MixerModel* model, int* d_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    // Embedding lookup.
    int threads = 256;
    int nblocks = (total + threads - 1) / threads;
    kernel_apply_embedding<<<nblocks, threads>>>(d_input_tokens, model->embedding_weight, model->embeddings,
                                                 batch, seq, embed, model->vocab_size);
    
    // Copy embeddings to first block_outputs buffer.
    int tensor_elements = total * embed;
    CHECK_CUDA(cudaMemcpy(model->block_outputs, model->embeddings, tensor_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Forward through each MixerBlock.
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        mixer_block_forward(handle, model->blocks[i], input_ptr, output_ptr, batch);
    }
    
    // Output projection: logits = final_output * (out_proj_weight)^T.
    float* final_output = model->block_outputs + model->num_layers * tensor_elements;
    int combined_batch_seq = batch * seq;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                model->vocab_size, combined_batch_seq, embed,
                &alpha,
                model->out_proj_weight, embed,
                final_output, embed,
                &beta,
                model->logits, model->vocab_size));
}

////////////////////////////////////////////////////////////////////////////////
// Compute cross-entropy loss and gradients directly on GPU
float compute_loss_and_gradients_gpu(MixerModel* model, int* d_target_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int vocab = model->vocab_size;
    
    // Launch kernel to compute softmax, loss, and gradients in one go
    int threads = 256;
    int blocks = (batch * seq + threads - 1) / threads;
    kernel_softmax_cross_entropy<<<blocks, threads>>>(
        model->logits, d_target_tokens,
        model->d_logits, model->d_loss_buffer,
        batch, seq, vocab
    );
    
    // Retrieve loss values
    CHECK_CUDA(cudaMemcpy(model->h_loss_buffer, model->d_loss_buffer, batch * seq * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum up the loss on CPU
    float total_loss = 0.0f;
    for (int i = 0; i < batch * seq; i++) {
        total_loss += model->h_loss_buffer[i];
    }
    
    return total_loss / (batch * seq);
}

//
// Backward pass through the output projection layer.
void output_projection_backward(cublasHandle_t handle, MixerModel* model) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    int combined_batch_seq = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    float* final_output = model->block_outputs + model->num_layers * (batch * seq * embed);
    float* d_final_output = model->d_block_outputs + model->num_layers * (batch * seq * embed);
    CHECK_CUDA(cudaMemset(d_final_output, 0, combined_batch_seq * embed * sizeof(float)));
    
    // Gradient w.r.t. output projection weights:
    // We want: out_proj_weight_grad = (d_logits)^T * final_output.
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                embed, vocab, combined_batch_seq,
                &alpha,
                final_output, embed,
                model->d_logits, vocab,
                &beta,
                model->out_proj_weight_grad, embed));
    
    // Gradient w.r.t. final_output: d_final_output = d_logits * out_proj_weight.
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                embed, combined_batch_seq, vocab,
                &alpha,
                model->out_proj_weight, embed,
                model->d_logits, vocab,
                &beta,
                d_final_output, embed));
}

//
// Backward pass through one MixerBlock.
void mixer_block_backward(cublasHandle_t handle, MixerBlock* block, float* d_output, float* d_input, int batch_size) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int total = batch_size * seq * embed;
    int total_trans = batch_size * embed * seq;
    int combined_batch = batch_size * seq;
    int combined_batch_embed = batch_size * embed;
    int threads = 256;
    int nblocks;
    float alpha = 1.0f, beta = 0.0f;
    
    // --- Channel Mixing Backward ---
    CHECK_CUDA(cudaMemcpy(block->d_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(block->d_channel_activated, block->d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    nblocks = (total + threads - 1) / threads;
    kernel_silu_deriv_mult<<<nblocks, threads>>>(block->channel_mixed, block->d_channel_activated, block->d_channel_mixed, total);

    // Gradient w.r.t. channel mixing weights
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                embed, embed, combined_batch,
                &alpha,
                block->d_channel_mixed, embed,
                block->residual, embed,
                &beta,
                block->channel_mixing_weight_grad, embed));
    
    // Gradient w.r.t. inputs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                embed, combined_batch, embed,
                &alpha,
                block->channel_mixing_weight, embed,
                block->d_channel_mixed, embed,
                &beta,
                d_input, embed));
    
    // Add d_output to d_input
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->d_output, 1, d_input, 1));
    
    // --- Token Mixing Backward ---
    CHECK_CUDA(cudaMemcpy(block->d_output, d_input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    nblocks = (total_trans + threads - 1) / threads;
    kernel_transpose_BSE_to_BES<<<nblocks, threads>>>(d_input, block->d_output_transposed, batch_size, seq, embed);

    nblocks = (total_trans + threads - 1) / threads;
    kernel_silu_deriv_mult<<<nblocks, threads>>>(block->token_mixed, block->d_output_transposed, block->d_token_mixed, total_trans);

    // Gradient w.r.t. token mixing weights
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                seq, seq, combined_batch_embed,
                &alpha,
                block->transposed, seq,
                block->d_token_mixed, seq,
                &beta,
                block->temp_grad, seq));
    
    nblocks = ((seq * seq) + threads - 1) / threads;
    kernel_apply_mask<<<nblocks, threads>>>(block->temp_grad, block->token_mixing_mask, block->temp_grad, seq * seq);

    // Add temp_grad to token_mixing_weight_grad
    CHECK_CUBLAS(cublasSaxpy(handle, seq * seq, &alpha, block->temp_grad, 1, block->token_mixing_weight_grad, 1));
    
    // Gradient w.r.t. input from token mixing
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                seq, combined_batch_embed, seq,
                &alpha,
                block->masked_weights, seq,
                block->d_token_mixed, seq,
                &beta,
                block->d_input_transposed, seq));
    
    nblocks = (total + threads - 1) / threads;
    kernel_transpose_BES_to_BSE<<<nblocks, threads>>>(block->d_input_transposed, d_input, batch_size, embed, seq);

    // Add d_output to d_input
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->d_output, 1, d_input, 1));
}

// CUDA kernel for embedding backward
__global__ void kernel_embedding_backward(const int* input_tokens, const float* d_embeddings,
                                        float* embedding_grad, int batch_size, int seq_length,
                                        int embed_dim, int vocab_size) {
    // Use atomicAdd for safe concurrent updates to the same embedding vectors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length) {
        int b = idx / seq_length;
        int s = idx % seq_length;
        int token = input_tokens[b * seq_length + s];
        if (token >= 0 && token < vocab_size) {
            for (int e = 0; e < embed_dim; e++) {
                atomicAdd(&embedding_grad[token * embed_dim + e], 
                         d_embeddings[b * seq_length * embed_dim + s * embed_dim + e]);
            }
        }
    }
}

// GPU-based embedding backward
void embedding_backward_gpu(MixerModel* model, int* d_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    
    // Zero out the embedding gradient first
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, model->vocab_size * embed * sizeof(float)));
    
    // Launch kernel to compute gradients
    int threads = 256;
    int blocks = (batch * seq + threads - 1) / threads;
    kernel_embedding_backward<<<blocks, threads>>>(
        d_input_tokens, model->d_block_outputs,
        model->embedding_weight_grad, batch, seq, embed, model->vocab_size
    );
}

//
// Backward pass through the entire model.
void mixer_model_backward(cublasHandle_t handle, MixerModel* model) {
    // Compute loss gradients (already done in compute_loss_and_gradients_gpu)
    output_projection_backward(handle, model);
    
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq * embed;
    
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output = model->d_block_outputs + (i + 1) * total;
        float* d_input = model->d_block_outputs + i * total;
        mixer_block_backward(handle, model->blocks[i], d_output, d_input, batch);
    }
    
    embedding_backward_gpu(model, model->d_input_tokens);
}

//
// Update weights using AdamW optimizer - GPU based version
void update_weights_adamw_gpu(MixerModel* model, float learning_rate) {
    model->t++;  // Increment time step.
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    float scale = model->batch_size * model->seq_length; // For normalization
    
    // Update embedding weights
    int threads = 256;
    int blocks;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    
    blocks = (vocab * embed + threads - 1) / threads;
    kernel_adam_update<<<blocks, threads>>>(
        model->embedding_weight, model->embedding_weight_grad,
        model->embedding_m, model->embedding_v,
        alpha_t, model->beta1, model->beta2, model->epsilon,
        model->weight_decay, learning_rate, vocab * embed, scale
    );
    
    // Update output projection weights
    blocks = (vocab * embed + threads - 1) / threads;
    kernel_adam_update<<<blocks, threads>>>(
        model->out_proj_weight, model->out_proj_weight_grad,
        model->out_proj_m, model->out_proj_v,
        alpha_t, model->beta1, model->beta2, model->epsilon,
        model->weight_decay, learning_rate, vocab * embed, scale
    );
    
    // Update MixerBlock weights
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int size_tok = block->seq_length * block->seq_length;
        int size_channel = block->embed_dim * block->embed_dim;
        
        // Token mixing weights
        blocks = (size_tok + threads - 1) / threads;
        kernel_adam_update<<<blocks, threads>>>(
            block->token_mixing_weight, block->token_mixing_weight_grad,
            block->token_mixing_m, block->token_mixing_v,
            alpha_t, model->beta1, model->beta2, model->epsilon,
            model->weight_decay, learning_rate, size_tok, scale
        );
        
        // Update masked weights - important to preserve the masking structure
        kernel_apply_mask<<<blocks, threads>>>(
            block->token_mixing_weight, block->token_mixing_mask, 
            block->masked_weights, size_tok
        );
        
        // Channel mixing weights
        blocks = (size_channel + threads - 1) / threads;
        kernel_adam_update<<<blocks, threads>>>(
            block->channel_mixing_weight, block->channel_mixing_weight_grad,
            block->channel_mixing_m, block->channel_mixing_v,
            alpha_t, model->beta1, model->beta2, model->epsilon,
            model->weight_decay, learning_rate, size_channel, scale
        );
    }
}

//
// Zero out all gradients (call this after every update).
void zero_gradients(MixerModel* model) {
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t size = vocab * embed * sizeof(float);
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, size));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, size));
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        size_t size_tok = block->seq_length * block->seq_length * sizeof(float);
        size_t size_chan = block->embed_dim * block->embed_dim * sizeof(float);
        CHECK_CUDA(cudaMemset(block->token_mixing_weight_grad, 0, size_tok));
        CHECK_CUDA(cudaMemset(block->channel_mixing_weight_grad, 0, size_chan));
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU softmax and sampling for text generation.
void softmax_cpu(const float* input, float* output, int size, float temperature) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf((input[i] - max_val) / temperature);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

int sample_from_distribution_cpu(const float* probs, int size) {
    float r = ((float)rand()) / ((float)RAND_MAX);
    float cdf = 0.0f;
    for (int i = 0; i < size; i++) {
        cdf += probs[i];
        if(r < cdf) return i;
    }
    return size - 1;
}

////////////////////////////////////////////////////////////////////////////////
// Encode a UTF-8 string into tokens where each token is made of two bytes.
int encode_string(const char* input, int* tokens, int max_tokens) {
    int len = 0;
    while (*input && len < max_tokens) {
        unsigned char first = (unsigned char)*input;
        input++;
        unsigned char second = 0;
        if (*input) {
            second = (unsigned char)*input;
            input++;
        }
        tokens[len++] = ((int)first << 8) | second;
    }
    return len;
}

// Decode tokens (each token represents two bytes) into a UTF-8 string.
void decode_tokens(const int* tokens, int num_tokens, char* output, int max_length) {
    int len = 0;
    for (int i = 0; i < num_tokens && len < max_length - 2; i++) {
        unsigned char first = (tokens[i] >> 8) & 0xFF;
        unsigned char second = tokens[i] & 0xFF;
        output[len++] = (char)first;
        if(len < max_length - 1) {
            output[len++] = (char)second;
        } else {
            break;
        }
    }
    output[len] = '\0';
}

////////////////////////////////////////////////////////////////////////////////
// Generate text from the model.
void generate_text(cublasHandle_t handle, MixerModel* model, const char* seed_text, int max_length, char* output, int output_size) {
    int* h_seed_tokens = (int*)malloc(model->seq_length * sizeof(int));
    int* h_input_batch = (int*)malloc(model->batch_size * model->seq_length * sizeof(int));
    float* h_probs = (float*)malloc(model->vocab_size * sizeof(float));
    
    memset(h_seed_tokens, 0, model->seq_length * sizeof(int));
    int seed_length = encode_string(seed_text, h_seed_tokens, model->seq_length);
    // If the seed produces fewer tokens than needed, shift the tokens right
    if (seed_length < model->seq_length) {
        memmove(h_seed_tokens + (model->seq_length - seed_length), h_seed_tokens, seed_length * sizeof(int));
        memset(h_seed_tokens, 0, (model->seq_length - seed_length) * sizeof(int));
    }
    for (int i = 0; i < model->seq_length; i++) {
        h_input_batch[i] = h_seed_tokens[i];
    }
    for (int b = 1; b < model->batch_size; b++) {
        for (int s = 0; s < model->seq_length; s++) {
            h_input_batch[b * model->seq_length + s] = 0;
        }
    }
    
    int* generated_tokens = (int*)malloc(max_length * sizeof(int));
    
    int* d_input_batch;
    CHECK_CUDA(cudaMalloc(&d_input_batch, model->batch_size * model->seq_length * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input_batch, h_input_batch, model->batch_size * model->seq_length * sizeof(int), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < max_length; i++) {
        mixer_model_forward(handle, model, d_input_batch);
        
        // Get logits for the last token of the first example.
        int seq = model->seq_length;
        int vocab = model->vocab_size;
        float* h_logits = (float*)malloc(vocab * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_logits, model->logits + ((0 * seq + (seq-1)) * vocab), vocab * sizeof(float), cudaMemcpyDeviceToHost));
        softmax_cpu(h_logits, h_probs, vocab, 0.7f);
        int next_token = sample_from_distribution_cpu(h_probs, vocab);
        generated_tokens[i] = next_token;
        free(h_logits);
        
        // Shift the first example left and append the new token.
        for (int s = 0; s < model->seq_length - 1; s++) {
            h_input_batch[s] = h_input_batch[s+1];
        }
        h_input_batch[model->seq_length - 1] = next_token;
        CHECK_CUDA(cudaMemcpy(d_input_batch, h_input_batch, model->batch_size * model->seq_length * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    decode_tokens(generated_tokens, max_length, output, output_size);
    
    free(h_seed_tokens);
    free(h_input_batch);
    free(h_probs);
    free(generated_tokens);
    CHECK_CUDA(cudaFree(d_input_batch));
}

////////////////////////////////////////////////////////////////////////////////
// Count model parameters.
int count_parameters(MixerModel* model) {
    int total_params = 0;
    total_params += model->vocab_size * model->embed_dim;
    total_params += model->vocab_size * model->embed_dim;
    for (int i = 0; i < model->num_layers; i++) {
        total_params += model->seq_length * model->seq_length;
        total_params += model->embed_dim * model->embed_dim;
    }
    return total_params;
}

////////////////////////////////////////////////////////////////////////////////
// Load text data from a file (CPU).
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
    if(bytes_read != file_size) {
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

////////////////////////////////////////////////////////////////////////////////
// Dataset and DataLoader implementation

Dataset* init_dataset(char* text, size_t text_size, int seq_length) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->text = text;
    dataset->text_size = text_size;
    dataset->seq_length = seq_length;
    
    // Compute number of tokens as floor(text_size/2)
    int num_tokens = (int)(text_size / 2);
    // Calculate number of valid starting indices (need seq_length+1 tokens for input+target)
    dataset->num_indices = num_tokens - seq_length;
    if (dataset->num_indices <= 0) {
        fprintf(stderr, "Error: Text is too short for the given sequence length (in token units)\n");
        free(dataset);
        return NULL;
    }
    
    // Allocate and fill indices array
    dataset->indices = (int*)malloc(dataset->num_indices * sizeof(int));
    for (int i = 0; i < dataset->num_indices; i++) {
        dataset->indices[i] = i;
    }
    
    // Allocate shuffled indices array and initialize
    dataset->shuffled_indices = (int*)malloc(dataset->num_indices * sizeof(int));
    memcpy(dataset->shuffled_indices, dataset->indices, dataset->num_indices * sizeof(int));
    dataset->current_position = 0;
    
    return dataset;
}

void free_dataset(Dataset* dataset) {
    // Don't free dataset->text as it's owned by the caller
    free(dataset->indices);
    free(dataset->shuffled_indices);
    free(dataset);
}

// Fisher-Yates shuffle algorithm
void shuffle_dataset(Dataset* dataset) {
    // Reset the position
    dataset->current_position = 0;
    
    // Copy from original indices
    memcpy(dataset->shuffled_indices, dataset->indices, dataset->num_indices * sizeof(int));
    
    // Shuffle the indices
    for (int i = dataset->num_indices - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = dataset->shuffled_indices[i];
        dataset->shuffled_indices[i] = dataset->shuffled_indices[j];
        dataset->shuffled_indices[j] = temp;
    }
}

// Get next batch from dataset.
int get_next_batch(Dataset* dataset, int batch_size, int* input_tokens, int* target_tokens) {
    // Check if we need to reshuffle
    if (dataset->current_position + batch_size > dataset->num_indices) {
        shuffle_dataset(dataset);
    }
    
    // Fill in the batch
    for (int b = 0; b < batch_size; b++) {
        int token_start = dataset->shuffled_indices[dataset->current_position++];
        // For each token position in our sequence, read two bytes to build the token.
        for (int s = 0; s < dataset->seq_length; s++) {
            int byte_index_input = (token_start + s) * 2;
            int byte_index_target = (token_start + s + 1) * 2;
            // Construct token from two bytes for input and target.
            input_tokens[b * dataset->seq_length + s] = (((unsigned char)dataset->text[byte_index_input]) << 8) | ((unsigned char)dataset->text[byte_index_input+1]);
            target_tokens[b * dataset->seq_length + s] = (((unsigned char)dataset->text[byte_index_target]) << 8) | ((unsigned char)dataset->text[byte_index_target+1]);
        }
    }
    
    return batch_size;
}

////////////////////////////////////////////////////////////////////////////////
// Save model to a binary file.
void save_model(MixerModel* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        return;
    }
    fwrite(&model->vocab_size, sizeof(int), 1, file);
    fwrite(&model->embed_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    fwrite(&model->batch_size, sizeof(int), 1, file);
    
    // Update host copies from device
    int vocab = model->vocab_size, embed = model->embed_dim;
    CHECK_CUDA(cudaMemcpy(model->h_embedding_weight, model->embedding_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(model->h_out_proj_weight, model->out_proj_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write from host copies
    fwrite(model->h_embedding_weight, sizeof(float), vocab * embed, file);
    fwrite(model->h_out_proj_weight, sizeof(float), vocab * embed, file);
    
    // For each mixer block, copy weights to host and write
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int seq = block->seq_length, embed = block->embed_dim;
        int size_tok = seq * seq;
        float* h_token_mixing_weight = (float*)malloc(size_tok * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_token_mixing_weight, block->token_mixing_weight, 
                   size_tok * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_token_mixing_weight, sizeof(float), size_tok, file);
        free(h_token_mixing_weight);
        
        int size_channel = embed * embed;
        float* h_channel_mixing_weight = (float*)malloc(size_channel * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_channel_mixing_weight, block->channel_mixing_weight, 
                   size_channel * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_channel_mixing_weight, sizeof(float), size_channel, file);
        free(h_channel_mixing_weight);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

////////////////////////////////////////////////////////////////////////////////
// Load model from a binary file.
MixerModel* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for reading\n", filename);
        return NULL;
    }
    int vocab_size, embed_dim, num_layers, seq_length, batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    
    // Read directly into host buffers
    fread(model->h_embedding_weight, sizeof(float), vocab_size * embed_dim, file);
    fread(model->h_out_proj_weight, sizeof(float), vocab_size * embed_dim, file);
    
    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(model->embedding_weight, model->h_embedding_weight, 
               vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(model->out_proj_weight, model->h_out_proj_weight, 
               vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // For each layer, read weights and upload to device
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int size_tok = seq_length * seq_length;
        float* h_token_mixing_weight = (float*)malloc(size_tok * sizeof(float));
        fread(h_token_mixing_weight, sizeof(float), size_tok, file);
        CHECK_CUDA(cudaMemcpy(block->token_mixing_weight, h_token_mixing_weight, 
                   size_tok * sizeof(float), cudaMemcpyHostToDevice));
                   
        // Update masked weights
        int threads = 256;
        int blocks = (size_tok + threads - 1) / threads;
        kernel_apply_mask<<<blocks, threads>>>(block->token_mixing_weight, 
                                               block->token_mixing_mask, 
                                               block->masked_weights, size_tok);
        free(h_token_mixing_weight);
        
        int size_channel = embed_dim * embed_dim;
        float* h_channel_mixing_weight = (float*)malloc(size_channel * sizeof(float));
        fread(h_channel_mixing_weight, sizeof(float), size_channel, file);
        CHECK_CUDA(cudaMemcpy(block->channel_mixing_weight, h_channel_mixing_weight, 
                   size_channel * sizeof(float), cudaMemcpyHostToDevice));
        free(h_channel_mixing_weight);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return model;
}

////////////////////////////////////////////////////////////////////////////////
// Main function.
int main(){
    srand(time(NULL));
    CHECK_CUDA(cudaSetDevice(0));
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    
    printf("Initializing Mixer model (CUDA version)...\n");
    int vocab_size = 65536;
    int embed_dim = 512;
    int num_layers = 16;
    int seq_length = 512;
    int batch_size = 32;
    
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    int num_params = count_parameters(model);
    printf("Model initialized with %d parameters\n", num_params);
    
    size_t text_size;
    char* text = load_text_file("../gutenberg_texts/combined_corpus.txt", &text_size);
    if (!text) {
        fprintf(stderr, "Failed to load text data\n");
        free_mixer_model(model);
        return 1;
    }
    printf("Loaded text corpus with %zu bytes\n", text_size);
    
    // Initialize dataset with the text corpus.
    Dataset* dataset = init_dataset(text, text_size, seq_length);
    if (!dataset) {
        fprintf(stderr, "Failed to initialize dataset\n");
        free(text);
        free_mixer_model(model);
        return 1;
    }
    printf("Dataset initialized with %d samples\n", dataset->num_indices);
    
    // Shuffle the dataset for training
    shuffle_dataset(dataset);
    
    float learning_rate = 0.0001f;
    int num_epochs = 10;
    
    // Calculate steps per epoch based on dataset size
    int steps_per_epoch = dataset->num_indices / batch_size;
    
    // Cap steps_per_epoch to a reasonable number if needed
    int max_steps_per_epoch = 10000;
    if (steps_per_epoch > max_steps_per_epoch) {
        printf("Limiting steps_per_epoch from %d to %d for efficiency\n", 
               steps_per_epoch, max_steps_per_epoch);
        steps_per_epoch = max_steps_per_epoch;
    }
    
    printf("Training for %d epochs with %d steps per epoch\n", num_epochs, steps_per_epoch);
    
    int* h_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    char seed_text[1024];
    size_t seed_pos = rand() % (text_size - 1024);
    strncpy(seed_text, text + seed_pos, 1023);
    seed_text[1023] = '\0';
    char generated_text[2048];
    
    printf("Starting training (CUDA version with DataLoader)...\n");
    time_t start_time = time(NULL);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Shuffle dataset at the beginning of each epoch
        shuffle_dataset(dataset);
        
        float epoch_loss = 0.0f;
        for (int step = 0; step < steps_per_epoch; step++) {
            // Get next batch from dataset
            get_next_batch(dataset, batch_size, h_input_tokens, h_target_tokens);
            
            // Copy input and target tokens to device
            CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_input_tokens, 
                       batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(model->d_target_tokens, h_target_tokens, 
                       batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
            
            // Forward pass
            mixer_model_forward(cublasHandle, model, model->d_input_tokens);
            
            // Compute loss and gradients in one step
            float step_loss = compute_loss_and_gradients_gpu(model, model->d_target_tokens);
            
            // Backward pass and update
            mixer_model_backward(cublasHandle, model);
            update_weights_adamw_gpu(model, learning_rate);
            
            // Zero gradients after update
            zero_gradients(model);
            
            epoch_loss += step_loss;
            
            if (step % 10 == 0) {
                printf("Epoch %d/%d, Step %d/%d, Loss: %.4f\n", 
                       epoch+1, num_epochs, step+1, steps_per_epoch, step_loss);
            }
            if(step % 100 == 0) {
                printf("Generating sample text...\n");
                printf("Sample seed text:\n%s\n\n", seed_text);
                generate_text(cublasHandle, model, seed_text, 100, generated_text, sizeof(generated_text));
                printf("Generated text:\n%s\n\n", generated_text);
            }
        }
        epoch_loss /= steps_per_epoch;
        time_t current_time = time(NULL);
        printf("\nEpoch %d/%d completed, Average Loss: %.4f, Time elapsed: %ld seconds\n\n", 
               epoch+1, num_epochs, epoch_loss, current_time - start_time);
    }
    
    save_model(model, "mixer_model_final_cuda.bin");
    
    free(h_input_tokens);
    free(h_target_tokens);
    free_dataset(dataset);
    free(text);
    free_mixer_model(model);
    
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    printf("Training completed!\n");
    return 0;
}