#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_SEQ_LENGTH 1024
#define MAX_VOCAB_SIZE 256
#define EMBED_DIM 512
#define NUM_LAYERS 8
#define BATCH_SIZE 64
#define TEMPERATURE 0.7f

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

////////////////////////////////////////////////////////////////////////////////
// Device helper functions and CUDA kernels

__device__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float device_silu(float x) {
    float sig = device_sigmoid(x);
    return x * sig;
}

__device__ float device_silu_derivative(float x) {
    float sig = device_sigmoid(x);
    return sig + x * sig * (1.0f - sig);
}

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

// Kernel: add vector b into a: a[i] += b[i]
__global__ void kernel_add(float* a, const float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        a[idx] += b[idx];
    }
}

// Kernel: device–device copy from src to dst
__global__ void kernel_copy(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        dst[idx] = src[idx];
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

////////////////////////////////////////////////////////////////////////////////
// Row–major GEMM wrapper using cuBLAS.
void gemm_rm(cublasHandle_t handle, int m, int n, int k,
             const float *A, const float *B, float *C,
             bool transA, bool transB) {
    float alpha = 1.0f, beta = 0.0f;
    // In this wrapper, the convention is as follows:
    // To compute C = A * B in row–major order (where A is m x k, B is k x n) you should call:
    //    gemm_rm(handle, m, n, k, A, B, C, true, false);
    // This gives no transpose on A (transA=true => opA = CUBLAS_OP_N) and a transpose on B (transB=false => opB = CUBLAS_OP_T),
    // so that cuBLAS computes: C = (B^T) * A in column–major which is equivalent to A * B in row–major.
    cublasOperation_t opA = transA ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = transB ? CUBLAS_OP_N : CUBLAS_OP_T;
    int lda = transA ? k : m;
    int ldb = transB ? n : k;
    // Note: Since cuBLAS assumes column–major storage, we swap the order of multiplication.
    cublasSgemm(handle, opB, opA,
                n, m, k,
                &alpha,
                B, ldb,
                A, lda,
                &beta,
                C, n);
}

////////////////////////////////////////////////////////////////////////////////
// Initialization and free functions for MixerBlock and MixerModel.

MixerBlock* init_mixer_block(int embed_dim, int seq_length) {
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
    cudaMalloc((void**)&block->token_mixing_weight, size_tok);
    cudaMemcpy(block->token_mixing_weight, h_token_mixing_weight, size_tok, cudaMemcpyHostToDevice);
    free(h_token_mixing_weight);
    
    cudaMalloc((void**)&block->token_mixing_weight_grad, size_tok);
    cudaMemset(block->token_mixing_weight_grad, 0, size_tok);
    cudaMalloc((void**)&block->token_mixing_m, size_tok);
    cudaMemset(block->token_mixing_m, 0, size_tok);
    cudaMalloc((void**)&block->token_mixing_v, size_tok);
    cudaMemset(block->token_mixing_v, 0, size_tok);
    
    // Create token mixing mask on host then copy.
    float* h_token_mixing_mask = (float*)malloc(size_tok);
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            h_token_mixing_mask[i * seq_length + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
    cudaMalloc((void**)&block->token_mixing_mask, size_tok);
    cudaMemcpy(block->token_mixing_mask, h_token_mixing_mask, size_tok, cudaMemcpyHostToDevice);
    free(h_token_mixing_mask);
    
    // Channel mixing weights.
    size_t size_channel = embed_dim * embed_dim * sizeof(float);
    float* h_channel_mixing_weight = (float*)malloc(size_channel);
    float scale_channel = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < embed_dim * embed_dim; i++){
        h_channel_mixing_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_channel);
    }
    cudaMalloc((void**)&block->channel_mixing_weight, size_channel);
    cudaMemcpy(block->channel_mixing_weight, h_channel_mixing_weight, size_channel, cudaMemcpyHostToDevice);
    free(h_channel_mixing_weight);
    
    cudaMalloc((void**)&block->channel_mixing_weight_grad, size_channel);
    cudaMemset(block->channel_mixing_weight_grad, 0, size_channel);
    cudaMalloc((void**)&block->channel_mixing_m, size_channel);
    cudaMemset(block->channel_mixing_m, 0, size_channel);
    cudaMalloc((void**)&block->channel_mixing_v, size_channel);
    cudaMemset(block->channel_mixing_v, 0, size_channel);
    
    // Allocate forward pass buffers.
    size_t tensor_size = BATCH_SIZE * seq_length * embed_dim * sizeof(float);
    cudaMalloc((void**)&block->input_buffer, tensor_size);
    cudaMalloc((void**)&block->residual, tensor_size);
    size_t tensor_size_trans = BATCH_SIZE * embed_dim * seq_length * sizeof(float);
    cudaMalloc((void**)&block->transposed, tensor_size_trans);
    cudaMalloc((void**)&block->token_mixed, tensor_size_trans);
    cudaMalloc((void**)&block->token_mix_activated, tensor_size_trans);
    cudaMalloc((void**)&block->channel_mixed, tensor_size);
    cudaMalloc((void**)&block->channel_mix_activated, tensor_size);
    
    // Allocate backward pass buffers.
    cudaMalloc((void**)&block->d_output, tensor_size);
    cudaMalloc((void**)&block->d_token_mixed, tensor_size_trans);
    cudaMalloc((void**)&block->d_channel_mixed, tensor_size);
    cudaMalloc((void**)&block->d_input, tensor_size);
    
    // Additional buffers.
    cudaMalloc((void**)&block->masked_weights, size_tok);
    cudaMalloc((void**)&block->d_channel_activated, tensor_size);
    cudaMalloc((void**)&block->d_output_transposed, tensor_size_trans);
    cudaMalloc((void**)&block->d_input_transposed, tensor_size_trans);
    cudaMalloc((void**)&block->temp_grad, size_tok);
    cudaMemset(block->temp_grad, 0, size_tok);
    
    return block;
}

void free_mixer_block(MixerBlock* block) {
    cudaFree(block->token_mixing_weight);
    cudaFree(block->token_mixing_weight_grad);
    cudaFree(block->token_mixing_mask);
    cudaFree(block->token_mixing_m);
    cudaFree(block->token_mixing_v);
    
    cudaFree(block->channel_mixing_weight);
    cudaFree(block->channel_mixing_weight_grad);
    cudaFree(block->channel_mixing_m);
    cudaFree(block->channel_mixing_v);
    
    cudaFree(block->input_buffer);
    cudaFree(block->residual);
    cudaFree(block->transposed);
    cudaFree(block->token_mixed);
    cudaFree(block->token_mix_activated);
    cudaFree(block->channel_mixed);
    cudaFree(block->channel_mix_activated);
    
    cudaFree(block->d_output);
    cudaFree(block->d_token_mixed);
    cudaFree(block->d_channel_mixed);
    cudaFree(block->d_input);
    
    cudaFree(block->masked_weights);
    cudaFree(block->d_channel_activated);
    cudaFree(block->d_output_transposed);
    cudaFree(block->d_input_transposed);
    cudaFree(block->temp_grad);
    
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
    cudaMalloc((void**)&model->embedding_weight, embed_matrix_size);
    cudaMemcpy(model->embedding_weight, h_embedding_weight, embed_matrix_size, cudaMemcpyHostToDevice);
    free(h_embedding_weight);
    
    cudaMalloc((void**)&model->embedding_weight_grad, embed_matrix_size);
    cudaMemset(model->embedding_weight_grad, 0, embed_matrix_size);
    cudaMalloc((void**)&model->embedding_m, embed_matrix_size);
    cudaMemset(model->embedding_m, 0, embed_matrix_size);
    cudaMalloc((void**)&model->embedding_v, embed_matrix_size);
    cudaMemset(model->embedding_v, 0, embed_matrix_size);
    
    // Output projection weights.
    cudaMalloc((void**)&model->out_proj_weight, embed_matrix_size);
    float* h_out_proj_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_out_proj_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    }
    cudaMemcpy(model->out_proj_weight, h_out_proj_weight, embed_matrix_size, cudaMemcpyHostToDevice);
    free(h_out_proj_weight);
    
    cudaMalloc((void**)&model->out_proj_weight_grad, embed_matrix_size);
    cudaMemset(model->out_proj_weight_grad, 0, embed_matrix_size);
    cudaMalloc((void**)&model->out_proj_m, embed_matrix_size);
    cudaMemset(model->out_proj_m, 0, embed_matrix_size);
    cudaMalloc((void**)&model->out_proj_v, embed_matrix_size);
    cudaMemset(model->out_proj_v, 0, embed_matrix_size);
    
    // Allocate MixerBlocks.
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_mixer_block(embed_dim, seq_length);
    }
    
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    cudaMalloc((void**)&model->embeddings, tensor_size);
    // Allocate (num_layers+1) outputs concatenated.
    cudaMalloc((void**)&model->block_outputs, (num_layers+1) * tensor_size);
    // Logits: [batch, seq, vocab_size]
    size_t logits_size = batch_size * seq_length * vocab_size * sizeof(float);
    cudaMalloc((void**)&model->logits, logits_size);
    
    cudaMalloc((void**)&model->d_logits, logits_size);
    cudaMalloc((void**)&model->d_block_outputs, (num_layers+1) * tensor_size);
    
    // softmax_probs on CPU.
    model->softmax_probs = (float*)malloc(vocab_size * sizeof(float));
    
    return model;
}

void free_mixer_model(MixerModel* model) {
    cudaFree(model->embedding_weight);
    cudaFree(model->embedding_weight_grad);
    cudaFree(model->embedding_m);
    cudaFree(model->embedding_v);
    
    cudaFree(model->out_proj_weight);
    cudaFree(model->out_proj_weight_grad);
    cudaFree(model->out_proj_m);
    cudaFree(model->out_proj_v);
    
    for (int i = 0; i < model->num_layers; i++) {
        free_mixer_block(model->blocks[i]);
    }
    free(model->blocks);
    
    cudaFree(model->embeddings);
    cudaFree(model->block_outputs);
    cudaFree(model->logits);
    
    cudaFree(model->d_logits);
    cudaFree(model->d_block_outputs);
    
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
    
    // Save input for backward.
    cudaMemcpy(block->input_buffer, input, total * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(block->residual, input, total * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Transpose input: [batch, seq, embed] -> [batch, embed, seq]
    int threads = 256;
    int nthreads = total;
    int nblocks = (nthreads + threads - 1) / threads;
    kernel_transpose_BSE_to_BES<<<nblocks, threads>>>(input, block->transposed, batch_size, seq, embed);
    
    // Create masked token mixing weights.
    nthreads = seq * seq;
    nblocks = (nthreads + threads - 1) / threads;
    kernel_apply_mask<<<nblocks, threads>>>(block->token_mixing_weight, block->token_mixing_mask, block->masked_weights, seq * seq);
    
    // Token mixing GEMM:
    // Compute: token_mixed = transposed * (masked_weights)^T.
    int combined_batch_embed = batch_size * embed;
    gemm_rm(handle, combined_batch_embed, seq, seq, block->transposed, block->masked_weights, block->token_mixed, true, false);
    // Note: For forward, passing (transA=true, transB=false) makes opA = CUBLAS_OP_N and opB = CUBLAS_OP_T, yielding the desired product.
    
    // Apply SiLU activation.
    nthreads = total_trans;
    nblocks = (nthreads + threads - 1) / threads;
    kernel_silu<<<nblocks, threads>>>(block->token_mixed, block->token_mix_activated, total_trans);
    
    // Transpose back: [batch, embed, seq] -> [batch, seq, embed]
    nthreads = total;
    nblocks = (nthreads + threads - 1) / threads;
    kernel_transpose_BES_to_BSE<<<nblocks, threads>>>(block->token_mix_activated, output, batch_size, embed, seq);
    
    // Add residual.
    nthreads = total;
    nblocks = (nthreads + threads - 1) / threads;
    kernel_add<<<nblocks, threads>>>(output, block->residual, total);
    
    // --- Channel Mixing ---
    // Save current output as residual.
    cudaMemcpy(block->residual, output, total * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Channel mixing GEMM: channel_mixed = output * (channel_mixing_weight)^T.
    int combined_batch = batch_size * seq;
    gemm_rm(handle, combined_batch, embed, embed, output, block->channel_mixing_weight, block->channel_mixed, true, false);
    // Apply SiLU activation.
    nthreads = total;
    nblocks = (nthreads + threads - 1) / threads;
    kernel_silu<<<nblocks, threads>>>(block->channel_mixed, block->channel_mix_activated, total);
    // Copy activated output and add residual.
    cudaMemcpy(output, block->channel_mix_activated, total * sizeof(float), cudaMemcpyDeviceToDevice);
    kernel_add<<<nblocks, threads>>>(output, block->residual, total);
}

//
// Forward pass through the entire model.
void mixer_model_forward(cublasHandle_t handle, MixerModel* model, int* d_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq;
    
    // Embedding lookup.
    int threads = 256;
    int nblocks = (total + threads - 1) / threads;
    kernel_apply_embedding<<<nblocks, threads>>>(d_input_tokens, model->embedding_weight, model->embeddings,
                                                   batch, seq, embed, model->vocab_size);
    
    // Copy embeddings to first block_outputs buffer.
    int tensor_elements = total * embed;
    cudaMemcpy(model->block_outputs, model->embeddings, tensor_elements * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Forward through each MixerBlock.
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        mixer_block_forward(handle, model->blocks[i], input_ptr, output_ptr, batch);
    }
    
    // Output projection: logits = final_output * (out_proj_weight)^T.
    float* final_output = model->block_outputs + model->num_layers * tensor_elements;
    int combined_batch_seq = batch * seq;
    // For output projection we want no transpose on final_output and a transpose on out_proj_weight.
    gemm_rm(handle, combined_batch_seq, model->vocab_size, embed, final_output, model->out_proj_weight, model->logits, true, false);
}

////////////////////////////////////////////////////////////////////////////////
// On–CPU: Compute cross–entropy loss and gradients.
// This function copies logits from device to host, computes softmax and loss, and writes the gradient (softmax - one–hot) into model->d_logits.
float compute_loss_and_gradients(MixerModel* model, int* h_target_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int vocab = model->vocab_size;
    int total_logits = batch * seq * vocab;
    
    float* h_logits = (float*)malloc(total_logits * sizeof(float));
    cudaMemcpy(h_logits, model->logits, total_logits * sizeof(float), cudaMemcpyDeviceToHost);
    
    float* h_d_logits = (float*)malloc(total_logits * sizeof(float));
    memset(h_d_logits, 0, total_logits * sizeof(float));
    
    float total_loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            int idx = (b * seq + s) * vocab;
            float max_logit = h_logits[idx];
            for (int v = 1; v < vocab; v++) {
                if (h_logits[idx + v] > max_logit)
                    max_logit = h_logits[idx + v];
            }
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; v++) {
                model->softmax_probs[v] = expf(h_logits[idx + v] - max_logit);
                sum_exp += model->softmax_probs[v];
            }
            for (int v = 0; v < vocab; v++) {
                model->softmax_probs[v] /= sum_exp;
            }
            int target = h_target_tokens[b * seq + s];
            total_loss += -logf(model->softmax_probs[target] + 1e-10f);
            for (int v = 0; v < vocab; v++) {
                h_d_logits[idx + v] = model->softmax_probs[v];
            }
            h_d_logits[idx + target] -= 1.0f;
        }
    }
    cudaMemcpy(model->d_logits, h_d_logits, total_logits * sizeof(float), cudaMemcpyHostToDevice);
    
    free(h_logits);
    free(h_d_logits);
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
    
    float* final_output = model->block_outputs + model->num_layers * (batch * seq * embed);
    float* d_final_output = model->d_block_outputs + model->num_layers * (batch * seq * embed);
    cudaMemset(d_final_output, 0, combined_batch_seq * embed * sizeof(float));
    
    // Gradient w.r.t. output projection weights:
    // We want: out_proj_weight_grad = (d_logits)^T * final_output.
    // In our wrapper, use flags to mimic CblasTrans on d_logits and no transpose on final_output.
    gemm_rm(handle, vocab, embed, combined_batch_seq, model->d_logits, final_output, model->out_proj_weight_grad, false, true);
    // Gradient w.r.t. final_output: d_final_output = d_logits * out_proj_weight.
    // For no transposition on both, we pass true flags.
    gemm_rm(handle, combined_batch_seq, embed, vocab, model->d_logits, model->out_proj_weight, d_final_output, true, true);
}

//
// Backward pass through one MixerBlock.
// (Note: The backward steps here closely mirror the CPU version.)
void mixer_block_backward(cublasHandle_t handle, MixerBlock* block, float* d_output, float* d_input, int batch_size) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int total = batch_size * seq * embed;
    int total_trans = batch_size * embed * seq;
    int combined_batch = batch_size * seq;
    int combined_batch_embed = batch_size * embed;
    int threads = 256;
    int nblocks = (total + threads - 1) / threads;
    
    // --- Channel Mixing Backward ---
    cudaMemcpy(block->d_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(block->d_channel_activated, block->d_output, total * sizeof(float), cudaMemcpyDeviceToDevice);
    nblocks = (total + threads - 1) / threads;
    kernel_silu_deriv_mult<<<nblocks, threads>>>(block->channel_mixed, block->d_channel_activated, block->d_channel_mixed, total);
    
    // Gradient w.r.t. channel mixing weights: use (residual)^T * d_channel_mixed.
    // For CPU, that was CblasTrans on residual and no transpose on d_channel_mixed.
    // In our wrapper, pass transA=false, transB=true.
    gemm_rm(handle, embed, embed, combined_batch, block->residual, block->d_channel_mixed, block->channel_mixing_weight_grad, false, true);
    // Gradient w.r.t. inputs: d_input = d_channel_mixed * channel_mixing_weight.
    // No transposition on both: pass true, true.
    gemm_rm(handle, combined_batch, embed, embed, block->d_channel_mixed, block->channel_mixing_weight, d_input, true, true);
    kernel_add<<<nblocks, threads>>>(d_input, block->d_output, total);
    
    // --- Token Mixing Backward ---
    cudaMemcpy(block->d_output, d_input, total * sizeof(float), cudaMemcpyDeviceToDevice);
    nblocks = (total_trans + threads - 1) / threads;
    kernel_transpose_BSE_to_BES<<<nblocks, threads>>>(d_input, block->d_output_transposed, batch_size, seq, embed);
    nblocks = (total_trans + threads - 1) / threads;
    kernel_silu_deriv_mult<<<nblocks, threads>>>(block->token_mixed, block->d_output_transposed, block->d_token_mixed, total_trans);
    
    // Gradient w.r.t. token mixing weights:
    // We need to compute: temp_grad = (d_token_mixed)^T * transposed.
    // CPU version calls for CblasTrans on d_token_mixed and no transpose on transposed.
    // Here, call gemm_rm with transA = false (to get transpose) and transB = true (to use matrix as is).
    gemm_rm(handle, seq, seq, combined_batch_embed, block->d_token_mixed, block->transposed, block->temp_grad, false, true);
    nblocks = ((seq * seq) + threads - 1) / threads;
    kernel_apply_mask<<<nblocks, threads>>>(block->temp_grad, block->token_mixing_mask, block->temp_grad, seq * seq);
    kernel_add<<<nblocks, threads>>>(block->token_mixing_weight_grad, block->temp_grad, seq * seq);
    
    // Gradient w.r.t. input from token mixing:
    // We want: d_input_transposed = d_token_mixed * masked_weights (no transposition on either).
    // So pass transA = true and transB = true.
    gemm_rm(handle, combined_batch_embed, seq, seq, block->d_token_mixed, block->masked_weights, block->d_input_transposed, true, true);
    nblocks = (total + threads - 1) / threads;
    kernel_transpose_BES_to_BSE<<<nblocks, threads>>>(block->d_input_transposed, d_input, batch_size, embed, seq);
    kernel_add<<<nblocks, threads>>>(d_input, block->d_output, total);
}

//
// Backward pass through the embedding layer.
// For each input token, accumulate the corresponding gradient from the first block output.
void embedding_backward(MixerModel* model, int* h_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq * embed;
    float* h_d_embeddings = (float*)malloc(total * sizeof(float));
    cudaMemcpy(h_d_embeddings, model->d_block_outputs, total * sizeof(float), cudaMemcpyDeviceToHost);
    float* h_grad = (float*)calloc(model->vocab_size * embed, sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            int token = h_input_tokens[b * seq + s];
            for (int e = 0; e < embed; e++) {
                h_grad[token * embed + e] += h_d_embeddings[b * seq * embed + s * embed + e];
            }
        }
    }
    cudaMemcpy(model->embedding_weight_grad, h_grad, model->vocab_size * embed * sizeof(float), cudaMemcpyHostToDevice);
    free(h_grad);
    free(h_d_embeddings);
}

//
// Backward pass through the entire model.
void mixer_model_backward(cublasHandle_t handle, MixerModel* model, int* h_input_tokens, int* h_target_tokens) {
    // Compute loss and its gradients.
    compute_loss_and_gradients(model, h_target_tokens);
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
    embedding_backward(model, h_input_tokens);
}

//
// Update weights using AdamW optimizer.
// (For simplicity, gradients are copied back to the host, updated and then pushed back to device.)
void update_weights_adamw(MixerModel* model, float learning_rate) {
    model->t++;  // Increment time step.
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update embedding weights.
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    int num_elements = vocab * embed;
    float* h_grad = (float*)malloc(num_elements * sizeof(float));
    float* h_weight = (float*)malloc(num_elements * sizeof(float));
    float* h_m = (float*)malloc(num_elements * sizeof(float));
    float* h_v = (float*)malloc(num_elements * sizeof(float));
    cudaMemcpy(h_grad, model->embedding_weight_grad, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weight, model->embedding_weight, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, model->embedding_m, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, model->embedding_v, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elements; i++) {
        float grad = h_grad[i] / (model->batch_size * model->seq_length);
        h_m[i] = model->beta1 * h_m[i] + (1.0f - model->beta1) * grad;
        h_v[i] = model->beta2 * h_v[i] + (1.0f - model->beta2) * grad * grad;
        float update = alpha_t * h_m[i] / (sqrtf(h_v[i]) + model->epsilon);
        h_weight[i] = h_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
    }
    cudaMemcpy(model->embedding_weight, h_weight, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->embedding_m, h_m, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->embedding_v, h_v, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    free(h_grad); free(h_weight); free(h_m); free(h_v);
    
    // Update output projection weights.
    num_elements = vocab * embed;
    h_grad = (float*)malloc(num_elements * sizeof(float));
    h_weight = (float*)malloc(num_elements * sizeof(float));
    h_m = (float*)malloc(num_elements * sizeof(float));
    h_v = (float*)malloc(num_elements * sizeof(float));
    cudaMemcpy(h_grad, model->out_proj_weight_grad, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weight, model->out_proj_weight, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, model->out_proj_m, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, model->out_proj_v, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elements; i++) {
        float grad = h_grad[i] / (model->batch_size * model->seq_length);
        h_m[i] = model->beta1 * h_m[i] + (1.0f - model->beta1) * grad;
        h_v[i] = model->beta2 * h_v[i] + (1.0f - model->beta2) * grad * grad;
        float update = alpha_t * h_m[i] / (sqrtf(h_v[i]) + model->epsilon);
        h_weight[i] = h_weight[i] * (1.0f - learning_rate * model->weight_decay) - update;
    }
    cudaMemcpy(model->out_proj_weight, h_weight, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->out_proj_m, h_m, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model->out_proj_v, h_v, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    free(h_grad); free(h_weight); free(h_m); free(h_v);
    
    // Update MixerBlock weights.
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int size_tok = block->seq_length * block->seq_length;
        float* h_grad_tok = (float*)malloc(size_tok * sizeof(float));
        float* h_weight_tok = (float*)malloc(size_tok * sizeof(float));
        float* h_m_tok = (float*)malloc(size_tok * sizeof(float));
        float* h_v_tok = (float*)malloc(size_tok * sizeof(float));
        cudaMemcpy(h_grad_tok, block->token_mixing_weight_grad, size_tok * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_weight_tok, block->token_mixing_weight, size_tok * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_m_tok, block->token_mixing_m, size_tok * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v_tok, block->token_mixing_v, size_tok * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_tok; i++) {
            float grad = h_grad_tok[i] / (model->batch_size * model->seq_length);
            h_m_tok[i] = model->beta1 * h_m_tok[i] + (1.0f - model->beta1) * grad;
            h_v_tok[i] = model->beta2 * h_v_tok[i] + (1.0f - model->beta2) * grad * grad;
            float update = alpha_t * h_m_tok[i] / (sqrtf(h_v_tok[i]) + model->epsilon);
            h_weight_tok[i] = h_weight_tok[i] * (1.0f - learning_rate * model->weight_decay) - update;
        }
        cudaMemcpy(block->token_mixing_weight, h_weight_tok, size_tok * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(block->token_mixing_m, h_m_tok, size_tok * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(block->token_mixing_v, h_v_tok, size_tok * sizeof(float), cudaMemcpyHostToDevice);
        free(h_grad_tok); free(h_weight_tok); free(h_m_tok); free(h_v_tok);
        
        int size_channel = block->embed_dim * block->embed_dim;
        float* h_grad_chan = (float*)malloc(size_channel * sizeof(float));
        float* h_weight_chan = (float*)malloc(size_channel * sizeof(float));
        float* h_m_chan = (float*)malloc(size_channel * sizeof(float));
        float* h_v_chan = (float*)malloc(size_channel * sizeof(float));
        cudaMemcpy(h_grad_chan, block->channel_mixing_weight_grad, size_channel * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_weight_chan, block->channel_mixing_weight, size_channel * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_m_chan, block->channel_mixing_m, size_channel * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v_chan, block->channel_mixing_v, size_channel * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size_channel; i++) {
            float grad = h_grad_chan[i] / (model->batch_size * model->seq_length);
            h_m_chan[i] = model->beta1 * h_m_chan[i] + (1.0f - model->beta1) * grad;
            h_v_chan[i] = model->beta2 * h_v_chan[i] + (1.0f - model->beta2) * grad * grad;
            float update = alpha_t * h_m_chan[i] / (sqrtf(h_v_chan[i]) + model->epsilon);
            h_weight_chan[i] = h_weight_chan[i] * (1.0f - learning_rate * model->weight_decay) - update;
        }
        cudaMemcpy(block->channel_mixing_weight, h_weight_chan, size_channel * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(block->channel_mixing_m, h_m_chan, size_channel * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(block->channel_mixing_v, h_v_chan, size_channel * sizeof(float), cudaMemcpyHostToDevice);
        free(h_grad_chan); free(h_weight_chan); free(h_m_chan); free(h_v_chan);
    }
}

//
// Zero out all gradients (call this after every update).
void zero_gradients(MixerModel* model) {
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t size = vocab * embed * sizeof(float);
    cudaMemset(model->embedding_weight_grad, 0, size);
    cudaMemset(model->out_proj_weight_grad, 0, size);
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        size_t size_tok = block->seq_length * block->seq_length * sizeof(float);
        size_t size_chan = block->embed_dim * block->embed_dim * sizeof(float);
        cudaMemset(block->token_mixing_weight_grad, 0, size_tok);
        cudaMemset(block->channel_mixing_weight_grad, 0, size_chan);
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU softmax and sampling for text generation.
void softmax_cpu(const float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val)
            max_val = input[i];
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
// Encode a UTF-8 string to byte tokens.
int encode_string(const char* input, int* tokens, int max_tokens) {
    int len = 0;
    while (*input && len < max_tokens) {
        tokens[len++] = (unsigned char)(*input++);
    }
    return len;
}

// Decode byte tokens to a UTF-8 string.
void decode_tokens(const int* tokens, int num_tokens, char* output, int max_length) {
    int len = 0;
    for (int i = 0; i < num_tokens && len < max_length - 1; i++) {
        output[len++] = (char)(tokens[i] & 0xFF);
    }
    output[len] = '\0';
}

////////////////////////////////////////////////////////////////////////////////
// Generate text from the model.
// We run inference from a seed string, always using the logits for the last token
// of the first example in the batch.
void generate_text(cublasHandle_t handle, MixerModel* model, const char* seed_text, int max_length, char* output, int output_size) {
    int* h_seed_tokens = (int*)malloc(model->seq_length * sizeof(int));
    int* h_input_batch = (int*)malloc(model->batch_size * model->seq_length * sizeof(int));
    float* h_probs = (float*)malloc(model->vocab_size * sizeof(float));
    
    memset(h_seed_tokens, 0, model->seq_length * sizeof(int));
    int seed_length = encode_string(seed_text, h_seed_tokens, model->seq_length);
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
    cudaMalloc((void**)&d_input_batch, model->batch_size * model->seq_length * sizeof(int));
    cudaMemcpy(d_input_batch, h_input_batch, model->batch_size * model->seq_length * sizeof(int), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < max_length; i++) {
        mixer_model_forward(handle, model, d_input_batch);
        // Get logits for the last token of the first example.
        int seq = model->seq_length;
        int vocab = model->vocab_size;
        float* h_logits = (float*)malloc(vocab * sizeof(float));
        cudaDeviceSynchronize();
        cudaMemcpy(h_logits, model->logits + ((0 * seq + (seq-1)) * vocab), vocab * sizeof(float), cudaMemcpyDeviceToHost);
        softmax_cpu(h_logits, h_probs, vocab);
        int next_token = sample_from_distribution_cpu(h_probs, vocab);
        generated_tokens[i] = next_token;
        free(h_logits);
        
        // Shift the first example left and append the new token.
        for (int s = 0; s < model->seq_length - 1; s++) {
            h_input_batch[s] = h_input_batch[s+1];
        }
        h_input_batch[model->seq_length - 1] = next_token;
        cudaMemcpy(d_input_batch, h_input_batch, model->batch_size * model->seq_length * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    decode_tokens(generated_tokens, max_length, output, output_size);
    
    free(h_seed_tokens);
    free(h_input_batch);
    free(h_probs);
    free(generated_tokens);
    cudaFree(d_input_batch);
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
// Create a batch of training samples (CPU).
void create_batch(char* text, size_t text_size, int seq_length, int batch_size, int* input_tokens, int* target_tokens) {
    for (int b = 0; b < batch_size; b++) {
        size_t start_pos = rand() % (text_size - seq_length - 1);
        for (int s = 0; s < seq_length; s++) {
            int token = (unsigned char)text[start_pos + s];
            input_tokens[b * seq_length + s] = token;
            target_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s + 1];
        }
    }
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
    
    int vocab = model->vocab_size, embed = model->embed_dim;
    float* h_embedding_weight = (float*)malloc(vocab * embed * sizeof(float));
    cudaMemcpy(h_embedding_weight, model->embedding_weight, vocab * embed * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_embedding_weight, sizeof(float), vocab * embed, file);
    free(h_embedding_weight);
    
    float* h_out_proj_weight = (float*)malloc(vocab * embed * sizeof(float));
    cudaMemcpy(h_out_proj_weight, model->out_proj_weight, vocab * embed * sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(h_out_proj_weight, sizeof(float), vocab * embed, file);
    free(h_out_proj_weight);
    
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int seq = block->seq_length, embed = block->embed_dim;
        int size_tok = seq * seq;
        float* h_token_mixing_weight = (float*)malloc(size_tok * sizeof(float));
        cudaMemcpy(h_token_mixing_weight, block->token_mixing_weight, size_tok * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(h_token_mixing_weight, sizeof(float), size_tok, file);
        free(h_token_mixing_weight);
        
        int size_channel = embed * embed;
        float* h_channel_mixing_weight = (float*)malloc(size_channel * sizeof(float));
        cudaMemcpy(h_channel_mixing_weight, block->channel_mixing_weight, size_channel * sizeof(float), cudaMemcpyDeviceToHost);
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
    
    int vocab = model->vocab_size, embed = model->embed_dim;
    float* h_embedding_weight = (float*)malloc(vocab * embed * sizeof(float));
    fread(h_embedding_weight, sizeof(float), vocab * embed, file);
    cudaMemcpy(model->embedding_weight, h_embedding_weight, vocab * embed * sizeof(float), cudaMemcpyHostToDevice);
    free(h_embedding_weight);
    
    float* h_out_proj_weight = (float*)malloc(vocab * embed * sizeof(float));
    fread(h_out_proj_weight, sizeof(float), vocab * embed, file);
    cudaMemcpy(model->out_proj_weight, h_out_proj_weight, vocab * embed * sizeof(float), cudaMemcpyHostToDevice);
    free(h_out_proj_weight);
    
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int size_tok = seq_length * seq_length;
        float* h_token_mixing_weight = (float*)malloc(size_tok * sizeof(float));
        fread(h_token_mixing_weight, sizeof(float), size_tok, file);
        cudaMemcpy(block->token_mixing_weight, h_token_mixing_weight, size_tok * sizeof(float), cudaMemcpyHostToDevice);
        free(h_token_mixing_weight);
        
        int size_channel = embed_dim * embed_dim;
        float* h_channel_mixing_weight = (float*)malloc(size_channel * sizeof(float));
        fread(h_channel_mixing_weight, sizeof(float), size_channel, file);
        cudaMemcpy(block->channel_mixing_weight, h_channel_mixing_weight, size_channel * sizeof(float), cudaMemcpyHostToDevice);
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
    cudaSetDevice(0);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    printf("Initializing Mixer model (CUDA version)...\n");
    int vocab_size = MAX_VOCAB_SIZE;
    int embed_dim = 512;
    int num_layers = 8;
    int seq_length = 1024;
    int batch_size = 64;
    
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
    
    float learning_rate = 0.0001f;
    int num_epochs = 10;
    int steps_per_epoch = text_size / (seq_length * batch_size);
    
    int* h_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    char seed_text[1024];
    size_t seed_pos = rand() % (text_size - 1024);
    strncpy(seed_text, text + seed_pos, 1023);
    seed_text[1023] = '\0';
    char generated_text[2048];
    
    printf("Starting training (CUDA version)...\n");
    time_t start_time = time(NULL);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        for (int step = 0; step < steps_per_epoch; step++) {
            create_batch(text, text_size, seq_length, batch_size, h_input_tokens, h_target_tokens);
            
            int* d_input_tokens;
            cudaMalloc((void**)&d_input_tokens, batch_size * seq_length * sizeof(int));
            cudaMemcpy(d_input_tokens, h_input_tokens, batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice);
            
            mixer_model_forward(cublasHandle, model, d_input_tokens);
            mixer_model_backward(cublasHandle, model, h_input_tokens, h_target_tokens);
            update_weights_adamw(model, learning_rate);
            // Zero gradients after update.
            zero_gradients(model);
            
            float step_loss = compute_loss_and_gradients(model, h_target_tokens);
            epoch_loss += step_loss;
            
            if (step % 10 == 0) {
                printf("Epoch %d/%d, Step %d/%d, Loss: %.4f\n", 
                       epoch+1, num_epochs, step+1, steps_per_epoch, step_loss);
            }
            if(step % 100 == 0) {
                printf("Generating sample text periodically...\n");
                printf("Sample seed text:\n%s\n\n", seed_text);
                generate_text(cublasHandle, model, seed_text, 100, generated_text, sizeof(generated_text));
                printf("Generated text:\n%s\n\n", generated_text);
            }
            
            cudaFree(d_input_tokens);
        }
        epoch_loss /= steps_per_epoch;
        time_t current_time = time(NULL);
        printf("\nEpoch %d/%d completed, Average Loss: %.4f, Time elapsed: %ld seconds\n\n", 
               epoch+1, num_epochs, epoch_loss, current_time - start_time);
    }
    
    save_model(model, "mixer_model_final_cuda.bin");
    
    free(h_input_tokens);
    free(h_target_tokens);
    free(text);
    free_mixer_model(model);
    
    cublasDestroy(cublasHandle);
    printf("Training completed!\n");
    return 0;
}