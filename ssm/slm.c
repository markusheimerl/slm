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

// SSMBlock structure - all pointers refer to device memory
typedef struct {
    // State transition parameters
    float* A;               // state_dim x state_dim (state transition matrix)
    float* A_grad;
    float* A_m;
    float* A_v;
    
    float* B;               // state_dim x embed_dim (input projection)
    float* B_grad;
    float* B_m;
    float* B_v;
    
    float* C;               // embed_dim x state_dim (output projection)
    float* C_grad;
    float* C_m;
    float* C_v;
    
    float* D;               // embed_dim x embed_dim (skip connection)
    float* D_grad;
    float* D_m;
    float* D_v;
    
    // RMSNorm parameters
    float* rmsnorm_weight;  // [embed_dim]
    float* rmsnorm_weight_grad;
    float* rmsnorm_m;
    float* rmsnorm_v;
    
    // Forward pass buffers
    float* input_buffer;    // [batch, seq, embed]
    float* normalized;      // [batch, seq, embed]
    float* states;          // [batch, seq, state_dim]
    float* next_states;     // [batch, seq, state_dim]
    float* temp_states;     // [batch, seq, state_dim]
    float* activated_states;// [batch, seq, state_dim]
    float* C_output;        // [batch, seq, embed]
    float* D_output;        // [batch, seq, embed]
    float* output;          // [batch, seq, embed]
    
    // Backward pass buffers
    float* d_output;        // [batch, seq, embed]
    float* d_input;         // [batch, seq, embed]
    float* d_normalized;    // [batch, seq, embed]
    float* d_states;        // [batch, seq, state_dim]
    float* d_activated_states; // [batch, seq, state_dim]
    float* d_C_output;      // [batch, seq, embed]
    float* d_D_output;      // [batch, seq, embed]
    
    // Additional buffers
    float* A_stable;        // [state_dim, state_dim] Stabilized A matrix
    float* rms_vars;        // [batch, seq] for RMSNorm variance caching
    
    // Dimensions
    int state_dim;          // SSM state dimension
    int embed_dim;          // Embedding dimension
    int seq_length;         // Sequence length
} SSMBlock;

// SSMModel structure
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
    
    // Array of SSMBlocks
    SSMBlock** blocks;
    
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
    
    // Persistent CPU buffer for loss computation
    float* h_loss_buffer;
    
    // Adam optimizer parameters
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;  // time step counter
    
    // Dimensions/hyperparameters
    int vocab_size;
    int embed_dim;
    int state_dim;
    int num_layers;
    int seq_length;
    int batch_size;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} SSMModel;

// Kernel: elementwise SiLU activation: out[i] = x * sigmoid(x)
__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

// Kernel: multiply gradient by SiLU derivative: grad_out = grad_in * silu_deriv(pre[i])
__global__ void silu_deriv_mult_kernel(const float* pre, const float* grad_in, float* grad_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = pre[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        float deriv = sig + x * sig * (1.0f - sig);
        grad_out[idx] = grad_in[idx] * deriv;
    }
}

// Kernel for RMSNorm forward pass
__global__ void rmsnorm_forward_kernel(const float* input, float* output, float* vars, 
                                      const float* weight, int batch_size, int seq_length, 
                                      int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length) {
        int b = idx / seq_length;
        int s = idx % seq_length;
        
        // Calculate variance
        float sum_squares = 0.0f;
        for (int e = 0; e < embed_dim; e++) {
            float val = input[(b * seq_length + s) * embed_dim + e];
            sum_squares += val * val;
        }
        float rms = sqrtf(sum_squares / embed_dim + 1e-8f);
        vars[idx] = rms;  // Save for backward pass
        
        // Normalize and scale
        for (int e = 0; e < embed_dim; e++) {
            int i = (b * seq_length + s) * embed_dim + e;
            output[i] = (input[i] / rms) * weight[e];
        }
    }
}

// Kernel for RMSNorm backward pass
__global__ void rmsnorm_backward_kernel(const float* input, const float* grad_out, 
                                      const float* vars, const float* weight,
                                      float* grad_in, float* grad_weight,
                                      int batch_size, int seq_length, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_length) {
        int b = idx / seq_length;
        int s = idx % seq_length;
        int base_idx = (b * seq_length + s) * embed_dim;
        float rms = vars[idx];
        float rms_inv = 1.0f / rms;
        
        // Calculate intermediate values for gradient computation
        float sum_grad_times_input = 0.0f;
        for (int e = 0; e < embed_dim; e++) {
            sum_grad_times_input += grad_out[base_idx + e] * weight[e] * input[base_idx + e];
        }
        
        float factor = -sum_grad_times_input / (embed_dim * rms * rms * rms);
        
        // Calculate gradients
        for (int e = 0; e < embed_dim; e++) {
            int i = base_idx + e;
            // Gradient for input
            grad_in[i] = grad_out[i] * weight[e] * rms_inv + factor * input[i];
            
            // Gradient for weight (will be atomically added later)
            atomicAdd(&grad_weight[e], grad_out[i] * input[i] * rms_inv);
        }
    }
}

// Compute stabilized A matrix to ensure eigenvalues are in unit disk
__global__ void compute_stable_A_kernel(float* A_stable, const float* A, int state_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < state_dim * state_dim) {
        int row = idx / state_dim;
        int col = idx % state_dim;
        
        if (row == col) {
            // Diagonal elements: scaled tanh for eigenvalue control
            A_stable[idx] = 0.9f * tanhf(A[idx]);
        } else {
            // Off-diagonal elements: scaled by matrix size
            A_stable[idx] = A[idx] / sqrtf((float)state_dim);
        }
    }
}

// Compute gradient of A from gradient of stabilized A
__global__ void compute_A_grad_from_stable_grad_kernel(float* A_grad, const float* A_stable_grad, 
                                                     const float* A, int state_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < state_dim * state_dim) {
        int row = idx / state_dim;
        int col = idx % state_dim;
        
        if (row == col) {
            // Diagonal derivative: d(tanh)/dA = sech²(A)
            float tanh_val = tanhf(A[idx]);
            float sech_squared = 1.0f - tanh_val * tanh_val;
            A_grad[idx] = A_stable_grad[idx] * 0.9f * sech_squared;
        } else {
            // Off-diagonal derivative: 1/sqrt(state_dim)
            A_grad[idx] = A_stable_grad[idx] / sqrtf((float)state_dim);
        }
    }
}

// Kernel: embedding lookup. For each token in input, copy its embedding row into output
__global__ void apply_embedding_kernel(const int* input_tokens, const float* embedding_weight, float* embeddings,
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

// Kernel to compute softmax and cross-entropy loss
__global__ void softmax_cross_entropy_kernel(const float* logits, const int* targets,
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

// CUDA kernel for embedding backward
__global__ void embedding_backward_kernel(const int* input_tokens, const float* d_embeddings,
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

// Kernel for AdamW weight update
__global__ void adamw_update_kernel(float* weight, const float* grad, float* m, float* v,
                                   float beta1, float beta2, float epsilon, float learning_rate,
                                   float weight_decay, float alpha_t, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        float g = grad[idx] / scale;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Initialization and free functions for SSMBlock and SSMModel
SSMBlock* init_ssm_block(int embed_dim, int state_dim, int seq_length, int batch_size) {
    SSMBlock* block = (SSMBlock*)malloc(sizeof(SSMBlock));
    block->embed_dim = embed_dim;
    block->state_dim = state_dim;
    block->seq_length = seq_length;
    
    // Allocate state transition matrices
    size_t size_A = state_dim * state_dim * sizeof(float);
    size_t size_B = state_dim * embed_dim * sizeof(float);
    size_t size_C = embed_dim * state_dim * sizeof(float);
    size_t size_D = embed_dim * embed_dim * sizeof(float);
    
    // Initialize A with scaled random values
    float* h_A = (float*)malloc(size_A);
    float scale_A = 1.0f / sqrtf((float)state_dim);
    for (int i = 0; i < state_dim * state_dim; i++) {
        h_A[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_A);
    }
    CHECK_CUDA(cudaMalloc(&block->A, size_A));
    CHECK_CUDA(cudaMemcpy(block->A, h_A, size_A, cudaMemcpyHostToDevice));
    free(h_A);
    
    // Initialize B with scaled random values
    float* h_B = (float*)malloc(size_B);
    float scale_B = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < state_dim * embed_dim; i++) {
        h_B[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_B);
    }
    CHECK_CUDA(cudaMalloc(&block->B, size_B));
    CHECK_CUDA(cudaMemcpy(block->B, h_B, size_B, cudaMemcpyHostToDevice));
    free(h_B);
    
    // Initialize C with scaled random values
    float* h_C = (float*)malloc(size_C);
    float scale_C = 1.0f / sqrtf((float)state_dim);
    for (int i = 0; i < embed_dim * state_dim; i++) {
        h_C[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_C);
    }
    CHECK_CUDA(cudaMalloc(&block->C, size_C));
    CHECK_CUDA(cudaMemcpy(block->C, h_C, size_C, cudaMemcpyHostToDevice));
    free(h_C);
    
    // Initialize D with scaled random values (skip connection)
    float* h_D = (float*)malloc(size_D);
    float scale_D = 1.0f / sqrtf((float)embed_dim);
    for (int i = 0; i < embed_dim * embed_dim; i++) {
        h_D[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_D);
    }
    CHECK_CUDA(cudaMalloc(&block->D, size_D));
    CHECK_CUDA(cudaMemcpy(block->D, h_D, size_D, cudaMemcpyHostToDevice));
    free(h_D);
    
    // Allocate gradient and optimizer buffers
    CHECK_CUDA(cudaMalloc(&block->A_grad, size_A));
    CHECK_CUDA(cudaMemset(block->A_grad, 0, size_A));
    CHECK_CUDA(cudaMalloc(&block->A_m, size_A));
    CHECK_CUDA(cudaMemset(block->A_m, 0, size_A));
    CHECK_CUDA(cudaMalloc(&block->A_v, size_A));
    CHECK_CUDA(cudaMemset(block->A_v, 0, size_A));
    
    CHECK_CUDA(cudaMalloc(&block->B_grad, size_B));
    CHECK_CUDA(cudaMemset(block->B_grad, 0, size_B));
    CHECK_CUDA(cudaMalloc(&block->B_m, size_B));
    CHECK_CUDA(cudaMemset(block->B_m, 0, size_B));
    CHECK_CUDA(cudaMalloc(&block->B_v, size_B));
    CHECK_CUDA(cudaMemset(block->B_v, 0, size_B));
    
    CHECK_CUDA(cudaMalloc(&block->C_grad, size_C));
    CHECK_CUDA(cudaMemset(block->C_grad, 0, size_C));
    CHECK_CUDA(cudaMalloc(&block->C_m, size_C));
    CHECK_CUDA(cudaMemset(block->C_m, 0, size_C));
    CHECK_CUDA(cudaMalloc(&block->C_v, size_C));
    CHECK_CUDA(cudaMemset(block->C_v, 0, size_C));
    
    CHECK_CUDA(cudaMalloc(&block->D_grad, size_D));
    CHECK_CUDA(cudaMemset(block->D_grad, 0, size_D));
    CHECK_CUDA(cudaMalloc(&block->D_m, size_D));
    CHECK_CUDA(cudaMemset(block->D_m, 0, size_D));
    CHECK_CUDA(cudaMalloc(&block->D_v, size_D));
    CHECK_CUDA(cudaMemset(block->D_v, 0, size_D));
    
    // Initialize RMSNorm weights to 1.0
    size_t rmsnorm_size = embed_dim * sizeof(float);
    float* h_rmsnorm_weight = (float*)malloc(rmsnorm_size);
    for (int i = 0; i < embed_dim; i++) {
        h_rmsnorm_weight[i] = 1.0f;
    }
    
    CHECK_CUDA(cudaMalloc(&block->rmsnorm_weight, rmsnorm_size));
    CHECK_CUDA(cudaMemcpy(block->rmsnorm_weight, h_rmsnorm_weight, rmsnorm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm_weight_grad, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm_weight_grad, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm_m, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm_m, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm_v, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm_v, 0, rmsnorm_size));
    
    free(h_rmsnorm_weight);
    
    // Allocate forward pass buffers
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    size_t state_tensor_size = batch_size * seq_length * state_dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&block->input_buffer, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->normalized, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->states, state_tensor_size));
    CHECK_CUDA(cudaMemset(block->states, 0, state_tensor_size)); // Initialize states to zero
    CHECK_CUDA(cudaMalloc(&block->next_states, state_tensor_size));
    CHECK_CUDA(cudaMalloc(&block->temp_states, state_tensor_size));
    CHECK_CUDA(cudaMalloc(&block->activated_states, state_tensor_size));
    CHECK_CUDA(cudaMalloc(&block->C_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->D_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->output, tensor_size));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&block->d_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_input, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_normalized, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_states, state_tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_activated_states, state_tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_C_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_D_output, tensor_size));
    
    // Additional buffers
    CHECK_CUDA(cudaMalloc(&block->A_stable, size_A));
    CHECK_CUDA(cudaMalloc(&block->rms_vars, batch_size * seq_length * sizeof(float)));
    
    return block;
}

void free_ssm_block(SSMBlock* block) {
    // Free state transition matrices
    CHECK_CUDA(cudaFree(block->A));
    CHECK_CUDA(cudaFree(block->A_grad));
    CHECK_CUDA(cudaFree(block->A_m));
    CHECK_CUDA(cudaFree(block->A_v));
    
    CHECK_CUDA(cudaFree(block->B));
    CHECK_CUDA(cudaFree(block->B_grad));
    CHECK_CUDA(cudaFree(block->B_m));
    CHECK_CUDA(cudaFree(block->B_v));
    
    CHECK_CUDA(cudaFree(block->C));
    CHECK_CUDA(cudaFree(block->C_grad));
    CHECK_CUDA(cudaFree(block->C_m));
    CHECK_CUDA(cudaFree(block->C_v));
    
    CHECK_CUDA(cudaFree(block->D));
    CHECK_CUDA(cudaFree(block->D_grad));
    CHECK_CUDA(cudaFree(block->D_m));
    CHECK_CUDA(cudaFree(block->D_v));
    
    // Free RMSNorm parameters
    CHECK_CUDA(cudaFree(block->rmsnorm_weight));
    CHECK_CUDA(cudaFree(block->rmsnorm_weight_grad));
    CHECK_CUDA(cudaFree(block->rmsnorm_m));
    CHECK_CUDA(cudaFree(block->rmsnorm_v));
    
    // Free forward pass buffers
    CHECK_CUDA(cudaFree(block->input_buffer));
    CHECK_CUDA(cudaFree(block->normalized));
    CHECK_CUDA(cudaFree(block->states));
    CHECK_CUDA(cudaFree(block->next_states));
    CHECK_CUDA(cudaFree(block->temp_states));
    CHECK_CUDA(cudaFree(block->activated_states));
    CHECK_CUDA(cudaFree(block->C_output));
    CHECK_CUDA(cudaFree(block->D_output));
    CHECK_CUDA(cudaFree(block->output));
    
    // Free backward pass buffers
    CHECK_CUDA(cudaFree(block->d_output));
    CHECK_CUDA(cudaFree(block->d_input));
    CHECK_CUDA(cudaFree(block->d_normalized));
    CHECK_CUDA(cudaFree(block->d_states));
    CHECK_CUDA(cudaFree(block->d_activated_states));
    CHECK_CUDA(cudaFree(block->d_C_output));
    CHECK_CUDA(cudaFree(block->d_D_output));
    
    // Free additional buffers
    CHECK_CUDA(cudaFree(block->A_stable));
    CHECK_CUDA(cudaFree(block->rms_vars));
    
    free(block);
}

SSMModel* init_ssm_model(int vocab_size, int embed_dim, int state_dim, int num_layers, int seq_length, int batch_size) {
    SSMModel* model = (SSMModel*)malloc(sizeof(SSMModel));
    
    // Store dimensions
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->state_dim = state_dim;
    model->num_layers = num_layers;
    model->seq_length = seq_length;
    model->batch_size = batch_size;
    
    // Initialize Adam optimizer parameters
    model->beta1 = 0.9f;
    model->beta2 = 0.999f;
    model->epsilon = 1e-8f;
    model->weight_decay = 0.01f;
    model->t = 0;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&model->cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(model->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // Embedding parameters
    size_t embed_matrix_size = vocab_size * embed_dim * sizeof(float);
    float scale_embed = 1.0f / sqrtf((float)embed_dim);
    
    // Initialize embedding weights on host
    float* h_embedding_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_embedding_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    }
    
    // Allocate and initialize embedding weights
    CHECK_CUDA(cudaMalloc(&model->embedding_weight, embed_matrix_size));
    CHECK_CUDA(cudaMemcpy(model->embedding_weight, h_embedding_weight, embed_matrix_size, cudaMemcpyHostToDevice));
    free(h_embedding_weight);
    
    // Allocate gradient and optimizer state for embedding
    CHECK_CUDA(cudaMalloc(&model->embedding_weight_grad, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->embedding_m, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_m, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->embedding_v, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->embedding_v, 0, embed_matrix_size));
    
    // Output projection parameters
    // Initialize output projection weights on host
    float* h_out_proj_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_out_proj_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    }
    
    // Allocate and initialize output projection weights
    CHECK_CUDA(cudaMalloc(&model->out_proj_weight, embed_matrix_size));
    CHECK_CUDA(cudaMemcpy(model->out_proj_weight, h_out_proj_weight, embed_matrix_size, cudaMemcpyHostToDevice));
    free(h_out_proj_weight);
    
    // Allocate gradient and optimizer state for output projection
    CHECK_CUDA(cudaMalloc(&model->out_proj_weight_grad, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->out_proj_m, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_m, 0, embed_matrix_size));
    CHECK_CUDA(cudaMalloc(&model->out_proj_v, embed_matrix_size));
    CHECK_CUDA(cudaMemset(model->out_proj_v, 0, embed_matrix_size));
    
    // Initialize SSMBlocks
    model->blocks = (SSMBlock**)malloc(num_layers * sizeof(SSMBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_ssm_block(embed_dim, state_dim, seq_length, batch_size);
    }
    
    // Allocate forward pass buffers
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&model->embeddings, tensor_size));
    // Allocate (num_layers+1) outputs concatenated
    CHECK_CUDA(cudaMalloc(&model->block_outputs, (num_layers+1) * tensor_size));
    
    // Allocate logits buffer
    size_t logits_size = batch_size * seq_length * vocab_size * sizeof(float);
    CHECK_CUDA(cudaMalloc(&model->logits, logits_size));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&model->d_logits, logits_size));
    CHECK_CUDA(cudaMalloc(&model->d_block_outputs, (num_layers+1) * tensor_size));
    
    // Allocate persistent device buffers for training
    CHECK_CUDA(cudaMalloc(&model->d_input_tokens, batch_size * seq_length * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_target_tokens, batch_size * seq_length * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_loss_buffer, batch_size * seq_length * sizeof(float)));
    
    // Allocate persistent host buffer for loss computation
    model->h_loss_buffer = (float*)malloc(batch_size * seq_length * sizeof(float));
    
    return model;
}

void free_ssm_model(SSMModel* model) {
    // Free embedding parameters
    CHECK_CUDA(cudaFree(model->embedding_weight));
    CHECK_CUDA(cudaFree(model->embedding_weight_grad));
    CHECK_CUDA(cudaFree(model->embedding_m));
    CHECK_CUDA(cudaFree(model->embedding_v));
    
    // Free output projection parameters
    CHECK_CUDA(cudaFree(model->out_proj_weight));
    CHECK_CUDA(cudaFree(model->out_proj_weight_grad));
    CHECK_CUDA(cudaFree(model->out_proj_m));
    CHECK_CUDA(cudaFree(model->out_proj_v));
    
    // Free SSMBlocks
    for (int i = 0; i < model->num_layers; i++) {
        free_ssm_block(model->blocks[i]);
    }
    free(model->blocks);
    
    // Free forward pass buffers
    CHECK_CUDA(cudaFree(model->embeddings));
    CHECK_CUDA(cudaFree(model->block_outputs));
    CHECK_CUDA(cudaFree(model->logits));
    
    // Free backward pass buffers
    CHECK_CUDA(cudaFree(model->d_logits));
    CHECK_CUDA(cudaFree(model->d_block_outputs));
    
    // Free persistent buffers
    CHECK_CUDA(cudaFree(model->d_input_tokens));
    CHECK_CUDA(cudaFree(model->d_target_tokens));
    CHECK_CUDA(cudaFree(model->d_loss_buffer));
    free(model->h_loss_buffer);
    
    // Destroy cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(model->cublas_handle));
    
    free(model);
}

// Forward pass through one SSMBlock
void ssm_block_forward(SSMBlock* block, float* input, float* output, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int state = block->state_dim;
    int total = batch_size * seq * embed;
    int total_state = batch_size * seq * state;
    float alpha = 1.0f, beta = 0.0f;
    
    // Save input for backward
    CHECK_CUDA(cudaMemcpy(block->input_buffer, input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Apply RMSNorm
    int threads = 256;
    int nblocks = (batch_size * seq + threads - 1) / threads;
    rmsnorm_forward_kernel<<<nblocks, threads>>>(input, block->normalized, block->rms_vars, 
                                                block->rmsnorm_weight, batch_size, seq, embed);
    
    // --- SSM State Update ---
    
    // Compute A_stable from A
    nblocks = (state * state + threads - 1) / threads;
    compute_stable_A_kernel<<<nblocks, threads>>>(block->A_stable, block->A, state);
    
    // Compute next_states = A_stable * states
    // Note: cuBLAS expects column-major order by default, but we're using row-major
    // For row-major: C^T = B^T * A^T means (next_states)^T = (states)^T * (A_stable)^T
    // This translates to using the operations as (B, A) instead of (A, B)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size * seq, state, state,
                &alpha,
                block->states, batch_size * seq,
                block->A_stable, state,
                &beta,
                block->next_states, batch_size * seq));
    
    // Add input contribution: next_states += B * normalized
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size * seq, state, embed,
                &alpha,
                block->normalized, batch_size * seq,
                block->B, state,
                &alpha,  // Adding to existing next_states
                block->next_states, batch_size * seq));
    
    // Apply SiLU activation
    nblocks = (total_state + threads - 1) / threads;
    silu_kernel<<<nblocks, threads>>>(block->next_states, block->activated_states, total_state);
    
    // Compute C_output = activated_states * C^T
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size * seq, embed, state,
                &alpha,
                block->activated_states, batch_size * seq,
                block->C, embed,
                &beta,
                block->C_output, batch_size * seq));
    
    // Compute D_output = normalized * D^T
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size * seq, embed, embed,
                &alpha,
                block->normalized, batch_size * seq,
                block->D, embed,
                &beta,
                block->D_output, batch_size * seq));
    
    // Output = C_output + D_output
    CHECK_CUDA(cudaMemcpy(output, block->C_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->D_output, 1, output, 1));
    
    // Add residual
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, input, 1, output, 1));
    
    // Update states for next time step
    CHECK_CUDA(cudaMemcpy(block->states, block->activated_states, total_state * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Forward pass through the entire model
void ssm_model_forward(SSMModel* model, int* d_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    // Embedding lookup
    int threads = 256;
    int nblocks = (total + threads - 1) / threads;
    apply_embedding_kernel<<<nblocks, threads>>>(d_input_tokens, model->embedding_weight, model->embeddings,
                                                 batch, seq, embed, model->vocab_size);
    
    // Copy embeddings to first block_outputs buffer
    int tensor_elements = total * embed;
    CHECK_CUDA(cudaMemcpy(model->block_outputs, model->embeddings, tensor_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Forward through each SSMBlock
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        ssm_block_forward(model->blocks[i], input_ptr, output_ptr, batch, model->cublas_handle);
    }
    
    // Output projection: logits = final_output * (out_proj_weight)^T
    float* final_output = model->block_outputs + model->num_layers * tensor_elements;
    int combined_batch_seq = batch * seq;
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                model->vocab_size, combined_batch_seq, embed,
                &alpha,
                model->out_proj_weight, embed,
                final_output, embed,
                &beta,
                model->logits, model->vocab_size));
}

// Compute cross-entropy loss and gradients
float compute_loss_and_gradients(SSMModel* model, int* d_target_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int vocab = model->vocab_size;
    
    // Launch kernel to compute softmax, loss, and gradients in one go
    int threads = 256;
    int blocks = (batch * seq + threads - 1) / threads;
    softmax_cross_entropy_kernel<<<blocks, threads>>>(
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

// Backward pass through one SSMBlock
void ssm_block_backward(SSMBlock* block, float* d_output, float* d_input, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int state = block->state_dim;
    int total = batch_size * seq * embed;
    int total_state = batch_size * seq * state;
    float alpha = 1.0f, beta = 0.0f;
    int threads = 256;
    int nblocks;
    
    // Save incoming gradient for residual
    CHECK_CUDA(cudaMemcpy(block->d_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Gradient for C_output and D_output
    CHECK_CUDA(cudaMemcpy(block->d_C_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(block->d_D_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Gradient for D: D_grad = (d_D_output)^T * normalized
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                embed, embed, batch_size * seq,
                &alpha,
                block->d_D_output, batch_size * seq,
                block->normalized, batch_size * seq,
                &beta,
                block->D_grad, embed));
    
    // Gradient for activated_states: d_activated_states = d_C_output * C
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                batch_size * seq, state, embed,
                &alpha,
                block->d_C_output, batch_size * seq,
                block->C, embed,
                &beta,
                block->d_activated_states, batch_size * seq));
    
    // Gradient for C: C_grad = (d_C_output)^T * activated_states
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                embed, state, batch_size * seq,
                &alpha,
                block->d_C_output, batch_size * seq,
                block->activated_states, batch_size * seq,
                &beta,
                block->C_grad, embed));
    
    // Apply SiLU derivative to get d_states
    nblocks = (total_state + threads - 1) / threads;
    silu_deriv_mult_kernel<<<nblocks, threads>>>(block->next_states, block->d_activated_states, block->d_states, total_state);
    
    // Gradient for B: B_grad = (d_states)^T * normalized
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                state, embed, batch_size * seq,
                &alpha,
                block->d_states, batch_size * seq,
                block->normalized, batch_size * seq,
                &beta,
                block->B_grad, state));
    
    // Gradient for A_stable: temp = (d_states)^T * states
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                state, state, batch_size * seq,
                &alpha,
                block->d_states, batch_size * seq,
                block->states, batch_size * seq,
                &beta,
                block->A_stable, state));  // Reuse A_stable as temporary
    
    // Convert A_stable gradient to A gradient
    nblocks = (state * state + threads - 1) / threads;
    compute_A_grad_from_stable_grad_kernel<<<nblocks, threads>>>(block->A_grad, block->A_stable, block->A, state);
    
    // Gradient for normalized from D_output: d_normalized += d_D_output * D
    CHECK_CUDA(cudaMemset(block->d_normalized, 0, total * sizeof(float)));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                batch_size * seq, embed, embed,
                &alpha,
                block->d_D_output, batch_size * seq,
                block->D, embed,
                &beta,
                block->d_normalized, batch_size * seq));
    
    // Gradient for normalized from states: d_normalized += d_states * B
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                batch_size * seq, embed, state,
                &alpha,
                block->d_states, batch_size * seq,
                block->B, state,
                &alpha,  // add to existing gradient
                block->d_normalized, batch_size * seq));
    
    // Gradient through RMSNorm
    nblocks = (batch_size * seq + threads - 1) / threads;
    CHECK_CUDA(cudaMemset(block->rmsnorm_weight_grad, 0, embed * sizeof(float)));
    rmsnorm_backward_kernel<<<nblocks, threads>>>(
        block->input_buffer, block->d_normalized, block->rms_vars,
        block->rmsnorm_weight, d_input, block->rmsnorm_weight_grad,
        batch_size, seq, embed
    );
    
    // Add residual gradient
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->d_output, 1, d_input, 1));
}

// Backward pass through the entire model
void ssm_model_backward(SSMModel* model) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    int total = batch * seq * embed;
    int combined_batch_seq = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    // --- Output projection backward ---
    float* final_output = model->block_outputs + model->num_layers * total;
    float* d_final_output = model->d_block_outputs + model->num_layers * total;
    CHECK_CUDA(cudaMemset(d_final_output, 0, combined_batch_seq * embed * sizeof(float)));
    
    // Gradient w.r.t. output projection weights:
    // We want: out_proj_weight_grad = (d_logits)^T * final_output.
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                embed, vocab, combined_batch_seq,
                &alpha,
                final_output, embed,
                model->d_logits, vocab,
                &beta,
                model->out_proj_weight_grad, embed));
    
    // Gradient w.r.t. final_output: d_final_output = d_logits * out_proj_weight.
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                embed, combined_batch_seq, vocab,
                &alpha,
                model->out_proj_weight, embed,
                model->d_logits, vocab,
                &beta,
                d_final_output, embed));
    
    // --- SSM blocks backward ---
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output = model->d_block_outputs + (i + 1) * total;
        float* d_input = model->d_block_outputs + i * total;
        ssm_block_backward(model->blocks[i], d_output, d_input, batch, model->cublas_handle);
    }
    
    // --- Embedding backward ---
    // Zero out the embedding gradient first
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, model->vocab_size * embed * sizeof(float)));
    
    // Launch kernel to compute gradients
    int threads = 256;
    int blocks = (batch * seq + threads - 1) / threads;
    embedding_backward_kernel<<<blocks, threads>>>(
        model->d_input_tokens, model->d_block_outputs,
        model->embedding_weight_grad, batch, seq, embed, model->vocab_size
    );
}

// Update weights using AdamW optimizer
void update_weights_adamw(SSMModel* model, float learning_rate) {
    model->t++;  // Increment time step.
    
    // Calculate bias correction factors
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Scale factor for gradient normalization
    float scale = model->batch_size * model->seq_length;
    
    // Update embedding weights
    int threads = 256;
    int blocks;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    
    blocks = (vocab * embed + threads - 1) / threads;
    adamw_update_kernel<<<blocks, threads>>>(model->embedding_weight, model->embedding_weight_grad,
                                            model->embedding_m, model->embedding_v,
                                            model->beta1, model->beta2, model->epsilon,
                                            learning_rate, model->weight_decay, alpha_t,
                                            vocab * embed, scale);
    
    // Update output projection weights
    blocks = (vocab * embed + threads - 1) / threads;
    adamw_update_kernel<<<blocks, threads>>>(model->out_proj_weight, model->out_proj_weight_grad,
                                            model->out_proj_m, model->out_proj_v,
                                            model->beta1, model->beta2, model->epsilon,
                                            learning_rate, model->weight_decay, alpha_t,
                                            vocab * embed, scale);
    
    // Update SSMBlock weights
    for (int l = 0; l < model->num_layers; l++) {
        SSMBlock* block = model->blocks[l];
        int state = block->state_dim;
        
        // A matrix
        blocks = (state * state + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->A, block->A_grad,
                                                block->A_m, block->A_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                state * state, scale);
        
        // B matrix
        blocks = (state * embed + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->B, block->B_grad,
                                                block->B_m, block->B_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                state * embed, scale);
        
        // C matrix
        blocks = (embed * state + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->C, block->C_grad,
                                                block->C_m, block->C_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                embed * state, scale);
        
        // D matrix
        blocks = (embed * embed + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->D, block->D_grad,
                                                block->D_m, block->D_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                embed * embed, scale);
        
        // RMSNorm weights
        blocks = (embed + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->rmsnorm_weight, block->rmsnorm_weight_grad,
                                                block->rmsnorm_m, block->rmsnorm_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                embed, scale);
    }
}

// Zero out all gradients
void zero_gradients(SSMModel* model) {
    int embed = model->embed_dim;
    int state = model->state_dim;
    int vocab = model->vocab_size;
    size_t embed_size = vocab * embed * sizeof(float);
    
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, embed_size));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, embed_size));
    
    for (int l = 0; l < model->num_layers; l++) {
        SSMBlock* block = model->blocks[l];
        
        CHECK_CUDA(cudaMemset(block->A_grad, 0, state * state * sizeof(float)));
        CHECK_CUDA(cudaMemset(block->B_grad, 0, state * embed * sizeof(float)));
        CHECK_CUDA(cudaMemset(block->C_grad, 0, embed * state * sizeof(float)));
        CHECK_CUDA(cudaMemset(block->D_grad, 0, embed * embed * sizeof(float)));
        CHECK_CUDA(cudaMemset(block->rmsnorm_weight_grad, 0, embed * sizeof(float)));
    }
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

// Get a batch of randomly sampled sequences from the text
void get_random_batch(const char* text, size_t text_size, int batch_size, int seq_length, 
                      int* input_tokens, int* target_tokens) {
    // Make sure we can sample valid sequences (text_size must be at least seq_length+1)
    if (text_size <= (size_t)seq_length) {
        fprintf(stderr, "Error: Text size is too small for the sequence length\n");
        return;
    }
    
    // Generate random starting positions for each sequence in the batch
    for (int b = 0; b < batch_size; b++) {
        // Random start position (leave room for sequence + 1 target)
        int start_pos = rand() % (text_size - seq_length - 1);
        
        // Fill input and target tokens
        for (int s = 0; s < seq_length; s++) {
            input_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s];
            target_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s + 1];
        }
    }
}

// Count model parameters
int count_parameters(SSMModel* model) {
    int total_params = 0;
    total_params += model->vocab_size * model->embed_dim;  // Embedding
    total_params += model->vocab_size * model->embed_dim;  // Output projection
    
    for (int i = 0; i < model->num_layers; i++) {
        total_params += model->state_dim * model->state_dim;  // A matrix
        total_params += model->state_dim * model->embed_dim;  // B matrix
        total_params += model->embed_dim * model->state_dim;  // C matrix
        total_params += model->embed_dim * model->embed_dim;  // D matrix
        total_params += model->embed_dim;                     // RMSNorm
    }
    
    return total_params;
}

// Save model to a binary file
void save_model(SSMModel* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        return;
    }
    
    // Write model dimensions
    fwrite(&model->vocab_size, sizeof(int), 1, file);
    fwrite(&model->embed_dim, sizeof(int), 1, file);
    fwrite(&model->state_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    fwrite(&model->batch_size, sizeof(int), 1, file);
    
    // Allocate temporary host buffers for parameter transfer
    int vocab = model->vocab_size, embed = model->embed_dim, state = model->state_dim;
    float* h_embedding_weight = (float*)malloc(vocab * embed * sizeof(float));
    float* h_out_proj_weight = (float*)malloc(vocab * embed * sizeof(float));
    
    // Copy embedding weights from device to host and write to file
    CHECK_CUDA(cudaMemcpy(h_embedding_weight, model->embedding_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_embedding_weight, sizeof(float), vocab * embed, file);
    
    // Copy output projection weights from device to host and write to file
    CHECK_CUDA(cudaMemcpy(h_out_proj_weight, model->out_proj_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_out_proj_weight, sizeof(float), vocab * embed, file);
    
    // For each SSM block, copy weights to host and write
    for (int i = 0; i < model->num_layers; i++) {
        SSMBlock* block = model->blocks[i];
        
        // A matrix
        float* h_A = (float*)malloc(state * state * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_A, block->A, state * state * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_A, sizeof(float), state * state, file);
        free(h_A);
        
        // B matrix
        float* h_B = (float*)malloc(state * embed * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_B, block->B, state * embed * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_B, sizeof(float), state * embed, file);
        free(h_B);
        
        // C matrix
        float* h_C = (float*)malloc(embed * state * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_C, block->C, embed * state * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_C, sizeof(float), embed * state, file);
        free(h_C);
        
        // D matrix
        float* h_D = (float*)malloc(embed * embed * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_D, block->D, embed * embed * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_D, sizeof(float), embed * embed, file);
        free(h_D);
        
        // RMSNorm weights
        float* h_rmsnorm = (float*)malloc(embed * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_rmsnorm, block->rmsnorm_weight, embed * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_rmsnorm, sizeof(float), embed, file);
        free(h_rmsnorm);
    }
    
    // Free temporary host buffers
    free(h_embedding_weight);
    free(h_out_proj_weight);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model from a binary file
SSMModel* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for reading\n", filename);
        return NULL;
    }
    
    // Read model dimensions
    int vocab_size, embed_dim, state_dim, num_layers, seq_length, batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize a new model with read dimensions
    SSMModel* model = init_ssm_model(vocab_size, embed_dim, state_dim, num_layers, seq_length, batch_size);
    
    // Allocate temporary host buffers for parameter transfer
    float* h_embedding_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_out_proj_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    
    // Read embedding weights to host buffer
    fread(h_embedding_weight, sizeof(float), vocab_size * embed_dim, file);
    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(model->embedding_weight, h_embedding_weight, 
               vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Read output projection weights to host buffer
    fread(h_out_proj_weight, sizeof(float), vocab_size * embed_dim, file);
    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(model->out_proj_weight, h_out_proj_weight, 
               vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // For each layer, read weights and upload to device
    for (int i = 0; i < model->num_layers; i++) {
        SSMBlock* block = model->blocks[i];
        
        // A matrix
        float* h_A = (float*)malloc(state_dim * state_dim * sizeof(float));
        fread(h_A, sizeof(float), state_dim * state_dim, file);
        CHECK_CUDA(cudaMemcpy(block->A, h_A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_A);
        
        // B matrix
        float* h_B = (float*)malloc(state_dim * embed_dim * sizeof(float));
        fread(h_B, sizeof(float), state_dim * embed_dim, file);
        CHECK_CUDA(cudaMemcpy(block->B, h_B, state_dim * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_B);
        
        // C matrix
        float* h_C = (float*)malloc(embed_dim * state_dim * sizeof(float));
        fread(h_C, sizeof(float), embed_dim * state_dim, file);
        CHECK_CUDA(cudaMemcpy(block->C, h_C, embed_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_C);
        
        // D matrix
        float* h_D = (float*)malloc(embed_dim * embed_dim * sizeof(float));
        fread(h_D, sizeof(float), embed_dim * embed_dim, file);
        CHECK_CUDA(cudaMemcpy(block->D, h_D, embed_dim * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_D);
        
        // RMSNorm weights
        float* h_rmsnorm = (float*)malloc(embed_dim * sizeof(float));
        fread(h_rmsnorm, sizeof(float), embed_dim, file);
        CHECK_CUDA(cudaMemcpy(block->rmsnorm_weight, h_rmsnorm, embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        free(h_rmsnorm);
    }
    
    // Free temporary host buffers
    free(h_embedding_weight);
    free(h_out_proj_weight);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return model;
}

// Generate text from the model with temperature-based sampling
void generate_text(SSMModel* model, const char* corpus, size_t corpus_size, int max_new_tokens, float temperature) {
    int seq_length = model->seq_length;
    
    // Allocate buffers for tokens
    int* h_tokens = (int*)malloc(seq_length * sizeof(int));
    
    int start_pos = rand() % (corpus_size - seq_length);
    
    // Print the full initial context we're using
    for (int i = 0; i < seq_length; i++) {
        h_tokens[i] = (unsigned char)corpus[start_pos + i];
        printf("%c", h_tokens[i]);
    }

    // Copy tokens to device
    CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_tokens, seq_length * sizeof(int), cudaMemcpyHostToDevice));
    
    // Reset the states of all SSM blocks
    for (int i = 0; i < model->num_layers; i++) {
        CHECK_CUDA(cudaMemset(model->blocks[i]->states, 0, 
                 model->batch_size * seq_length * model->state_dim * sizeof(float)));
    }
    
    // Generate tokens one by one
    for (int i = 0; i < max_new_tokens; i++) {
        // Forward pass
        ssm_model_forward(model, model->d_input_tokens);
        
        // Get logits for the last token
        float* h_logits = (float*)malloc(model->vocab_size * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_logits, 
                   model->logits + (seq_length - 1) * model->vocab_size, 
                   model->vocab_size * sizeof(float), 
                   cudaMemcpyDeviceToHost));
        
        // Apply temperature with epsilon to prevent division by zero
        float temp = temperature + 1e-7f;  // Add small epsilon
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] /= temp;
        }
        
        // Softmax for probabilities
        float max_logit = h_logits[0];
        for (int v = 1; v < model->vocab_size; v++) {
            if (h_logits[v] > max_logit) {
                max_logit = h_logits[v];
            }
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] /= sum_exp;  // Now h_logits contains probabilities
        }
        
        // Sample from distribution
        float r = (float)rand() / (float)RAND_MAX;
        float cdf = 0.0f;
        int next_token = 0;
        for (int v = 0; v < model->vocab_size; v++) {
            cdf += h_logits[v];
            if (r < cdf) {
                next_token = v;
                break;
            }
        }
        
        // Print the generated character
        printf("%c", next_token);
        fflush(stdout);
        
        // Shift tokens to the left and add new token at the end
        for (int j = 0; j < seq_length - 1; j++) {
            h_tokens[j] = h_tokens[j + 1];
        }
        h_tokens[seq_length - 1] = next_token;
        
        // Copy updated tokens to device
        CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_tokens, seq_length * sizeof(int), cudaMemcpyHostToDevice));
        
        free(h_logits);
    }
    
    printf("\n");
    free(h_tokens);
}

// Main function
int main(int argc, char** argv) {
    srand(time(NULL));
    CHECK_CUDA(cudaSetDevice(0));
    
    int vocab_size = 256;
    int embed_dim = 1536;
    int state_dim = 256;     // SSM state dimension
    int num_layers = 8;
    int seq_length = 2048;
    int batch_size = 4;
    
    SSMModel* model;
    
    if (argc > 1) {
        // Load existing model
        model = load_model(argv[1]);
    } else {
        // Create new model
        model = init_ssm_model(vocab_size, embed_dim, state_dim, num_layers, seq_length, batch_size);
    }
    
    printf("Model with %d parameters\n", count_parameters(model));
    
    size_t text_size;
    char* text = load_text_file("../combined_corpus.txt", &text_size);
    printf("Loaded text corpus with %zu bytes\n", text_size);
    
    float learning_rate = 0.0001f;
    int total_training_steps = 100000;
    
    printf("Training for %d total steps with learning rate %.6f\n", 
           total_training_steps, learning_rate);
    
    int* h_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    time_t start_time = time(NULL);

    for (int step = 0; step < total_training_steps; step++) {
        // Reset SSM states at the start of each batch
        for (int i = 0; i < model->num_layers; i++) {
            CHECK_CUDA(cudaMemset(model->blocks[i]->states, 0, 
                     model->batch_size * seq_length * model->state_dim * sizeof(float)));
        }
        
        // Get random batch from text
        get_random_batch(text, text_size, batch_size, seq_length, h_input_tokens, h_target_tokens);
        
        // Copy input and target tokens to device
        CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_input_tokens, 
                   batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(model->d_target_tokens, h_target_tokens, 
                   batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
        
        // Forward pass
        ssm_model_forward(model, model->d_input_tokens);
        
        // Compute loss and gradients in one step
        float step_loss = compute_loss_and_gradients(model, model->d_target_tokens);
        
        // Backward pass and update
        ssm_model_backward(model);
        update_weights_adamw(model, learning_rate);
        
        // Zero gradients after update
        zero_gradients(model);
        
        if (step % 10 == 0) {
            printf("Step %d/%d, Loss: %.4f\n", step, total_training_steps, step_loss);
        }
        
        if (step % 10000 == 0 && step > 0) {
            time_t current_time = time(NULL);
            printf("\n======= Sample generation at step %d after %ld seconds =======\n", step, current_time - start_time);
            generate_text(model, text, text_size, 128, 0.8f);
            printf("\n");
        }
    }
    
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_ssm_model.bin", localtime(&now));
    save_model(model, model_fname);
    
    free(h_input_tokens);
    free(h_target_tokens);
    free(text);
    free_ssm_model(model);
    return 0;
}