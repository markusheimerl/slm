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
    // Dilated causal 1D convolution parameters
    float* conv_weight;           // [embed_dim x kernel_size]
    float* conv_weight_grad;
    float* conv_m;
    float* conv_v;
    int kernel_size;
    int dilation;
    
    // Channel mixing parameters
    float* channel_up_weight;     // [embed_dim x (embed_dim*4)]
    float* channel_up_weight_grad;
    float* channel_up_m;
    float* channel_up_v;
    
    float* channel_down_weight;   // [(embed_dim*4) x embed_dim]
    float* channel_down_weight_grad;
    float* channel_down_m;
    float* channel_down_v;
    
    // RMSNorm parameters
    float* rmsnorm1_weight;       // [embed_dim]
    float* rmsnorm1_weight_grad;
    float* rmsnorm1_m;
    float* rmsnorm1_v;
    
    float* rmsnorm2_weight;       // [embed_dim]
    float* rmsnorm2_weight_grad;
    float* rmsnorm2_m;
    float* rmsnorm2_v;
    
    // Forward pass buffers
    float* input_buffer;
    float* residual;
    float* normalized1;
    float* normalized2;
    float* conv_output;
    float* conv_activated;
    float* channel_up_output;
    float* channel_up_activated;
    float* channel_mixed;
    
    // Backward pass buffers
    float* d_output;
    float* d_conv_output;
    float* d_channel_mixed;
    float* d_channel_up_output;
    float* d_channel_up_activated;
    float* d_input;
    float* d_normalized1;
    float* d_normalized2;
    
    // Additional buffers
    float* rms_vars;              // [batch, seq] for RMSNorm variance caching
     
    // Dimensions
    int embed_dim;
    int hidden_dim;
    int seq_length;
} MixerBlock;

// MixerModel structure.
typedef struct {
    // Embedding parameters
    float* embedding_weight;      // [vocab_size x embed_dim]
    float* embedding_weight_grad;
    float* embedding_m;
    float* embedding_v;
    
    // Output projection parameters
    float* out_proj_weight;       // [vocab_size x embed_dim]
    float* out_proj_weight_grad;
    float* out_proj_m;
    float* out_proj_v;
    
    // Array of MixerBlocks
    MixerBlock** blocks;
    
    // Forward pass buffers
    float* embeddings;
    float* block_outputs;
    float* logits;
    
    // Backward pass buffers
    float* d_logits;
    float* d_block_outputs;
    
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
    int hidden_dim;
    int num_layers;
    int seq_length;
    int batch_size;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} MixerModel;

// CUDA Kernels

// Kernel: elementwise SiLU activation: out[i] = x * sigmoid(x)
__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

// Kernel: multiply gradient by SiLU derivative
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

// Kernel: Dilated causal convolution forward pass
__global__ void dilated_causal_conv1d_forward_kernel(const float* input, const float* weight, float* output,
                                                    int batch_size, int seq_length, int embed_dim, 
                                                    int kernel_size, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size * seq_length * embed_dim) {
        int b = idx / (seq_length * embed_dim);
        int rem = idx % (seq_length * embed_dim);
        int s = rem / embed_dim;
        int e = rem % embed_dim;
        
        float result = 0.0f;
        
        // For each position in the kernel
        for(int k = 0; k < kernel_size; k++) {
            // Apply dilation - look back dilation * k positions
            int pos = s - k * dilation;
            
            // Only use valid positions (causal constraint)
            if(pos >= 0) {
                float x = input[(b * seq_length + pos) * embed_dim + e];
                float w = weight[e * kernel_size + k];
                result += x * w;
            }
        }
        
        output[idx] = result;
    }
}

// Kernel: Dilated causal convolution backward pass for input gradient
__global__ void dilated_causal_conv1d_backward_input_kernel(const float* grad_out, const float* weight, float* grad_in,
                                                          int batch_size, int seq_length, int embed_dim, 
                                                          int kernel_size, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size * seq_length * embed_dim) {
        int b = idx / (seq_length * embed_dim);
        int rem = idx % (seq_length * embed_dim);
        int s = rem / embed_dim;
        int e = rem % embed_dim;
        
        float grad = 0.0f;
        
        // For each position in the kernel that would have used this input
        for(int k = 0; k < kernel_size; k++) {
            // Future position that would use this input with dilation
            int pos = s + k * dilation;
            
            // Only consider valid positions
            if(pos < seq_length) {
                float g = grad_out[(b * seq_length + pos) * embed_dim + e];
                float w = weight[e * kernel_size + k];
                grad += g * w;
            }
        }
        
        grad_in[idx] = grad;
    }
}

// Kernel: Dilated causal convolution backward pass for weight gradient
__global__ void dilated_causal_conv1d_backward_weight_kernel(const float* input, const float* grad_out, float* grad_weight,
                                                           int batch_size, int seq_length, int embed_dim, 
                                                           int kernel_size, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < embed_dim * kernel_size) {
        int e = idx / kernel_size;
        int k = idx % kernel_size;
        
        float grad = 0.0f;
        
        // Sum gradients over all batches and valid sequence positions
        for(int b = 0; b < batch_size; b++) {
            for(int s = 0; s < seq_length; s++) {
                // Input position with dilation
                int input_pos = s - k * dilation;
                
                // Only use valid positions (causal constraint)
                if(input_pos >= 0) {
                    float x = input[(b * seq_length + input_pos) * embed_dim + e];
                    float g = grad_out[(b * seq_length + s) * embed_dim + e];
                    grad += x * g;
                }
            }
        }
        
        grad_weight[idx] = grad;
    }
}

// Kernel: embedding lookup
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

// Initialization and free functions for MixerBlock and MixerModel
MixerBlock* init_mixer_block(int embed_dim, int seq_length, int batch_size, int layer_idx) {
    MixerBlock* block = (MixerBlock*)malloc(sizeof(MixerBlock));
    block->embed_dim = embed_dim;
    block->seq_length = seq_length;
    block->hidden_dim = embed_dim * 4;
    block->kernel_size = 8;
    block->dilation = 1 << (layer_idx % 8);
    
    // Initialize causal convolution weights
    size_t size_conv = embed_dim * block->kernel_size * sizeof(float);
    float scale_conv = 1.0f / sqrtf((float)block->kernel_size);
    
    // Initialize convolution weights on host
    float* h_conv_weight = (float*)malloc(size_conv);
    for (int i = 0; i < embed_dim * block->kernel_size; i++) {
        h_conv_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_conv);
    }
    
    // Allocate and initialize convolution weights
    CHECK_CUDA(cudaMalloc(&block->conv_weight, size_conv));
    CHECK_CUDA(cudaMemcpy(block->conv_weight, h_conv_weight, size_conv, cudaMemcpyHostToDevice));
    free(h_conv_weight);
    
    // Allocate gradient and optimizer state for convolution
    CHECK_CUDA(cudaMalloc(&block->conv_weight_grad, size_conv));
    CHECK_CUDA(cudaMemset(block->conv_weight_grad, 0, size_conv));
    CHECK_CUDA(cudaMalloc(&block->conv_m, size_conv));
    CHECK_CUDA(cudaMemset(block->conv_m, 0, size_conv));
    CHECK_CUDA(cudaMalloc(&block->conv_v, size_conv));
    CHECK_CUDA(cudaMemset(block->conv_v, 0, size_conv));
    
    // Channel mixing up-projection parameters
    size_t size_up_channel = embed_dim * block->hidden_dim * sizeof(float);
    float scale_up_channel = 1.0f / sqrtf((float)embed_dim);
    
    // Initialize channel up-projection weights on host
    float* h_channel_up_weight = (float*)malloc(size_up_channel);
    for (int i = 0; i < embed_dim * block->hidden_dim; i++){
        h_channel_up_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_up_channel);
    }
    
    // Allocate and initialize channel up-projection weights
    CHECK_CUDA(cudaMalloc(&block->channel_up_weight, size_up_channel));
    CHECK_CUDA(cudaMemcpy(block->channel_up_weight, h_channel_up_weight, size_up_channel, cudaMemcpyHostToDevice));
    free(h_channel_up_weight);
    
    // Allocate gradient and optimizer state for channel up-projection
    CHECK_CUDA(cudaMalloc(&block->channel_up_weight_grad, size_up_channel));
    CHECK_CUDA(cudaMemset(block->channel_up_weight_grad, 0, size_up_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_up_m, size_up_channel));
    CHECK_CUDA(cudaMemset(block->channel_up_m, 0, size_up_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_up_v, size_up_channel));
    CHECK_CUDA(cudaMemset(block->channel_up_v, 0, size_up_channel));
    
    // Channel mixing down-projection parameters
    size_t size_down_channel = block->hidden_dim * embed_dim * sizeof(float);
    float scale_down_channel = 1.0f / sqrtf((float)block->hidden_dim);
    
    // Initialize channel down-projection weights on host
    float* h_channel_down_weight = (float*)malloc(size_down_channel);
    for (int i = 0; i < block->hidden_dim * embed_dim; i++){
        h_channel_down_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_down_channel);
    }
    
    // Allocate and initialize channel down-projection weights
    CHECK_CUDA(cudaMalloc(&block->channel_down_weight, size_down_channel));
    CHECK_CUDA(cudaMemcpy(block->channel_down_weight, h_channel_down_weight, size_down_channel, cudaMemcpyHostToDevice));
    free(h_channel_down_weight);
    
    // Allocate gradient and optimizer state for channel down-projection
    CHECK_CUDA(cudaMalloc(&block->channel_down_weight_grad, size_down_channel));
    CHECK_CUDA(cudaMemset(block->channel_down_weight_grad, 0, size_down_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_down_m, size_down_channel));
    CHECK_CUDA(cudaMemset(block->channel_down_m, 0, size_down_channel));
    CHECK_CUDA(cudaMalloc(&block->channel_down_v, size_down_channel));
    CHECK_CUDA(cudaMemset(block->channel_down_v, 0, size_down_channel));
    
    // Initialize RMSNorm weights (gamma) to 1.0
    size_t rmsnorm_size = embed_dim * sizeof(float);
    float* h_rmsnorm_weight = (float*)malloc(rmsnorm_size);
    for (int i = 0; i < embed_dim; i++) {
        h_rmsnorm_weight[i] = 1.0f;
    }
    
    // RMSNorm1 parameters
    CHECK_CUDA(cudaMalloc(&block->rmsnorm1_weight, rmsnorm_size));
    CHECK_CUDA(cudaMemcpy(block->rmsnorm1_weight, h_rmsnorm_weight, rmsnorm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm1_weight_grad, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm1_weight_grad, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm1_m, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm1_m, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm1_v, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm1_v, 0, rmsnorm_size));
    
    // RMSNorm2 parameters
    CHECK_CUDA(cudaMalloc(&block->rmsnorm2_weight, rmsnorm_size));
    CHECK_CUDA(cudaMemcpy(block->rmsnorm2_weight, h_rmsnorm_weight, rmsnorm_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm2_weight_grad, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm2_weight_grad, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm2_m, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm2_m, 0, rmsnorm_size));
    CHECK_CUDA(cudaMalloc(&block->rmsnorm2_v, rmsnorm_size));
    CHECK_CUDA(cudaMemset(block->rmsnorm2_v, 0, rmsnorm_size));
    
    free(h_rmsnorm_weight);
    
    // Allocate forward pass buffers
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    size_t tensor_size_hidden = batch_size * seq_length * block->hidden_dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&block->input_buffer, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->residual, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->normalized1, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->normalized2, tensor_size));
    
    CHECK_CUDA(cudaMalloc(&block->conv_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->conv_activated, tensor_size));
    
    // Buffers for up-projection and activation
    CHECK_CUDA(cudaMalloc(&block->channel_up_output, tensor_size_hidden));
    CHECK_CUDA(cudaMalloc(&block->channel_up_activated, tensor_size_hidden));
    CHECK_CUDA(cudaMalloc(&block->channel_mixed, tensor_size));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&block->d_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_conv_output, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_channel_mixed, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_channel_up_output, tensor_size_hidden));
    CHECK_CUDA(cudaMalloc(&block->d_channel_up_activated, tensor_size_hidden));
    CHECK_CUDA(cudaMalloc(&block->d_input, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_normalized1, tensor_size));
    CHECK_CUDA(cudaMalloc(&block->d_normalized2, tensor_size));
    
    // Allocate RMSNorm variance cache
    CHECK_CUDA(cudaMalloc(&block->rms_vars, batch_size * seq_length * sizeof(float)));
    
    return block;
}

void free_mixer_block(MixerBlock* block) {
    // Free convolution parameters
    CHECK_CUDA(cudaFree(block->conv_weight));
    CHECK_CUDA(cudaFree(block->conv_weight_grad));
    CHECK_CUDA(cudaFree(block->conv_m));
    CHECK_CUDA(cudaFree(block->conv_v));
    
    // Free channel mixing parameters - up projection
    CHECK_CUDA(cudaFree(block->channel_up_weight));
    CHECK_CUDA(cudaFree(block->channel_up_weight_grad));
    CHECK_CUDA(cudaFree(block->channel_up_m));
    CHECK_CUDA(cudaFree(block->channel_up_v));
    
    // Free channel mixing parameters - down projection
    CHECK_CUDA(cudaFree(block->channel_down_weight));
    CHECK_CUDA(cudaFree(block->channel_down_weight_grad));
    CHECK_CUDA(cudaFree(block->channel_down_m));
    CHECK_CUDA(cudaFree(block->channel_down_v));
    
    // Free RMSNorm parameters
    CHECK_CUDA(cudaFree(block->rmsnorm1_weight));
    CHECK_CUDA(cudaFree(block->rmsnorm1_weight_grad));
    CHECK_CUDA(cudaFree(block->rmsnorm1_m));
    CHECK_CUDA(cudaFree(block->rmsnorm1_v));
    
    CHECK_CUDA(cudaFree(block->rmsnorm2_weight));
    CHECK_CUDA(cudaFree(block->rmsnorm2_weight_grad));
    CHECK_CUDA(cudaFree(block->rmsnorm2_m));
    CHECK_CUDA(cudaFree(block->rmsnorm2_v));
    
    // Free forward pass buffers
    CHECK_CUDA(cudaFree(block->input_buffer));
    CHECK_CUDA(cudaFree(block->residual));
    CHECK_CUDA(cudaFree(block->normalized1));
    CHECK_CUDA(cudaFree(block->normalized2));
    CHECK_CUDA(cudaFree(block->conv_output));
    CHECK_CUDA(cudaFree(block->conv_activated));
    CHECK_CUDA(cudaFree(block->channel_up_output));
    CHECK_CUDA(cudaFree(block->channel_up_activated));
    CHECK_CUDA(cudaFree(block->channel_mixed));
    
    // Free backward pass buffers
    CHECK_CUDA(cudaFree(block->d_output));
    CHECK_CUDA(cudaFree(block->d_conv_output));
    CHECK_CUDA(cudaFree(block->d_channel_mixed));
    CHECK_CUDA(cudaFree(block->d_channel_up_output));
    CHECK_CUDA(cudaFree(block->d_channel_up_activated));
    CHECK_CUDA(cudaFree(block->d_input));
    CHECK_CUDA(cudaFree(block->d_normalized1));
    CHECK_CUDA(cudaFree(block->d_normalized2));
    
    // Free RMSNorm variance cache
    CHECK_CUDA(cudaFree(block->rms_vars));
    
    free(block);
}

MixerModel* init_mixer_model(int vocab_size, int embed_dim, int num_layers, int seq_length, int batch_size) {
    MixerModel* model = (MixerModel*)malloc(sizeof(MixerModel));
    
    // Store dimensions
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->hidden_dim = embed_dim * 4;
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
    
    // Initialize MixerBlocks
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_mixer_block(embed_dim, seq_length, batch_size, i);
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

void free_mixer_model(MixerModel* model) {
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
    
    // Free MixerBlocks
    for (int i = 0; i < model->num_layers; i++) {
        free_mixer_block(model->blocks[i]);
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

// Forward pass through one MixerBlock with dilated causal convolution
void mixer_block_forward(MixerBlock* block, float* input, float* output, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int hidden = block->hidden_dim;
    int total = batch_size * seq * embed;
    int total_hidden = batch_size * seq * hidden;
    float alpha = 1.0f, beta = 0.0f;
    
    // Save input for backward
    CHECK_CUDA(cudaMemcpy(block->input_buffer, input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Apply RMSNorm
    int threads = 256;
    int nblocks = (batch_size * seq + threads - 1) / threads;
    rmsnorm_forward_kernel<<<nblocks, threads>>>(input, block->normalized1, block->rms_vars, 
                                                block->rmsnorm1_weight, batch_size, seq, embed);
    
    // --- Dilated Causal 1D Convolution ---
    nblocks = (total + threads - 1) / threads;
    dilated_causal_conv1d_forward_kernel<<<nblocks, threads>>>(block->normalized1, block->conv_weight,
                                                              block->conv_output, batch_size, seq, embed, 
                                                              block->kernel_size, block->dilation);
    
    // Apply SiLU activation
    nblocks = (total + threads - 1) / threads;
    silu_kernel<<<nblocks, threads>>>(block->conv_output, block->conv_activated, total);
    
    // Add residual
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, input, 1, block->conv_activated, 1));
    CHECK_CUDA(cudaMemcpy(block->residual, block->conv_activated, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // --- Channel Mixing ---
    
    // Apply RMSNorm
    nblocks = (batch_size * seq + threads - 1) / threads;
    rmsnorm_forward_kernel<<<nblocks, threads>>>(block->residual, block->normalized2, block->rms_vars, 
                                                block->rmsnorm2_weight, batch_size, seq, embed);
    
    // Up-projection GEMM: channel_up_output = normalized2 * (channel_up_weight)^T
    int combined_batch = batch_size * seq;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                hidden, combined_batch, embed,
                &alpha,
                block->channel_up_weight, embed,
                block->normalized2, embed,
                &beta,
                block->channel_up_output, hidden));
    
    // Apply SiLU activation to up-projected output
    nblocks = (total_hidden + threads - 1) / threads;
    silu_kernel<<<nblocks, threads>>>(block->channel_up_output, block->channel_up_activated, total_hidden);
    
    // Down-projection GEMM: channel_mixed = channel_up_activated * (channel_down_weight)^T
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                embed, combined_batch, hidden,
                &alpha,
                block->channel_down_weight, hidden,
                block->channel_up_activated, hidden,
                &beta,
                output, embed));
    
    // Add residual
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->residual, 1, output, 1));
}

// Forward pass through the entire model.
void mixer_model_forward(MixerModel* model, int* d_input_tokens) {
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
    
    // Forward through each MixerBlock
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        mixer_block_forward(model->blocks[i], input_ptr, output_ptr, batch, model->cublas_handle);
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
float compute_loss_and_gradients(MixerModel* model, int* d_target_tokens) {
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

// Backward pass through one MixerBlock with dilated causal convolution.
void mixer_block_backward(MixerBlock* block, float* d_output, float* d_input, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int hidden = block->hidden_dim;
    int total = batch_size * seq * embed;
    int total_hidden = batch_size * seq * hidden;
    int combined_batch = batch_size * seq;
    int threads = 256;
    int nblocks;
    float alpha = 1.0f, beta = 0.0f;
    
    // --- Channel Mixing Backward ---
    
    // Save d_output for residual gradient
    CHECK_CUDA(cudaMemcpy(block->d_output, d_output, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Gradient w.r.t. down-projection weights
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                hidden, embed, combined_batch,
                &alpha,
                block->channel_up_activated, hidden,
                block->d_output, embed,
                &beta,
                block->channel_down_weight_grad, hidden));
    
    // Gradient w.r.t. channel_up_activated
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden, combined_batch, embed,
                &alpha,
                block->channel_down_weight, hidden,
                block->d_output, embed,
                &beta,
                block->d_channel_up_activated, hidden));
    
    // Gradient through SiLU activation
    nblocks = (total_hidden + threads - 1) / threads;
    silu_deriv_mult_kernel<<<nblocks, threads>>>(block->channel_up_output, block->d_channel_up_activated, 
                                               block->d_channel_up_output, total_hidden);
    
    // Gradient w.r.t. up-projection weights
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                embed, hidden, combined_batch,
                &alpha,
                block->normalized2, embed,
                block->d_channel_up_output, hidden,
                &beta,
                block->channel_up_weight_grad, embed));
    
    // Gradient w.r.t. normalized2 input
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                embed, combined_batch, hidden,
                &alpha,
                block->channel_up_weight, embed,
                block->d_channel_up_output, hidden,
                &beta,
                block->d_normalized2, embed));
    
    // Gradient through RMSNorm2
    nblocks = (batch_size * seq + threads - 1) / threads;
    CHECK_CUDA(cudaMemset(block->rmsnorm2_weight_grad, 0, embed * sizeof(float)));
    rmsnorm_backward_kernel<<<nblocks, threads>>>(
        block->residual, block->d_normalized2, block->rms_vars, 
        block->rmsnorm2_weight, d_input, block->rmsnorm2_weight_grad,
        batch_size, seq, embed
    );
    
    // Add d_output to d_input for residual connection
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->d_output, 1, d_input, 1));
    
    // --- Dilated Causal Convolution Backward ---
    
    // Save d_input for residual gradient
    CHECK_CUDA(cudaMemcpy(block->d_output, d_input, total * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Gradient through SiLU activation
    nblocks = (total + threads - 1) / threads;
    silu_deriv_mult_kernel<<<nblocks, threads>>>(block->conv_output, d_input, block->d_conv_output, total);
    
    // Gradient w.r.t. convolution weights
    nblocks = (embed * block->kernel_size + threads - 1) / threads;
    CHECK_CUDA(cudaMemset(block->conv_weight_grad, 0, embed * block->kernel_size * sizeof(float)));
    dilated_causal_conv1d_backward_weight_kernel<<<nblocks, threads>>>(
        block->normalized1, block->d_conv_output, block->conv_weight_grad,
        batch_size, seq, embed, block->kernel_size, block->dilation);
    
    // Gradient w.r.t. normalized1 input
    nblocks = (total + threads - 1) / threads;
    CHECK_CUDA(cudaMemset(block->d_normalized1, 0, total * sizeof(float)));
    dilated_causal_conv1d_backward_input_kernel<<<nblocks, threads>>>(
        block->d_conv_output, block->conv_weight, block->d_normalized1,
        batch_size, seq, embed, block->kernel_size, block->dilation);
    
    // Gradient through RMSNorm1
    nblocks = (batch_size * seq + threads - 1) / threads;
    CHECK_CUDA(cudaMemset(block->rmsnorm1_weight_grad, 0, embed * sizeof(float)));
    rmsnorm_backward_kernel<<<nblocks, threads>>>(
        block->input_buffer, block->d_normalized1, block->rms_vars, 
        block->rmsnorm1_weight, d_input, block->rmsnorm1_weight_grad,
        batch_size, seq, embed
    );
    
    // Add d_output to d_input for residual connection
    CHECK_CUBLAS(cublasSaxpy(handle, total, &alpha, block->d_output, 1, d_input, 1));
}

// Backward pass through the entire model.
void mixer_model_backward(MixerModel* model) {
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
    
    // Gradient w.r.t. output projection weights
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                embed, vocab, combined_batch_seq,
                &alpha,
                final_output, embed,
                model->d_logits, vocab,
                &beta,
                model->out_proj_weight_grad, embed));
    
    // Gradient w.r.t. final_output
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                embed, combined_batch_seq, vocab,
                &alpha,
                model->out_proj_weight, embed,
                model->d_logits, vocab,
                &beta,
                d_final_output, embed));
    
    // --- Mixer blocks backward ---
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output = model->d_block_outputs + (i + 1) * total;
        float* d_input = model->d_block_outputs + i * total;
        mixer_block_backward(model->blocks[i], d_output, d_input, batch, model->cublas_handle);
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
void update_weights_adamw(MixerModel* model, float learning_rate) {
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
    
    // Update MixerBlock weights
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int size_conv = block->embed_dim * block->kernel_size;
        int size_up = block->embed_dim * block->hidden_dim;
        int size_down = block->hidden_dim * block->embed_dim;
        
        // Convolution weights
        blocks = (size_conv + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->conv_weight, block->conv_weight_grad,
                                                block->conv_m, block->conv_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                size_conv, scale);
        
        // Channel up-projection weights
        blocks = (size_up + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->channel_up_weight, block->channel_up_weight_grad,
                                                block->channel_up_m, block->channel_up_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                size_up, scale);
        
        // Channel down-projection weights
        blocks = (size_down + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->channel_down_weight, block->channel_down_weight_grad,
                                                block->channel_down_m, block->channel_down_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                size_down, scale);
        
        // RMSNorm1 weights
        blocks = (embed + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->rmsnorm1_weight, block->rmsnorm1_weight_grad,
                                                block->rmsnorm1_m, block->rmsnorm1_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                embed, scale);
        
        // RMSNorm2 weights
        blocks = (embed + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->rmsnorm2_weight, block->rmsnorm2_weight_grad,
                                                block->rmsnorm2_m, block->rmsnorm2_v,
                                                model->beta1, model->beta2, model->epsilon,
                                                learning_rate, model->weight_decay, alpha_t,
                                                embed, scale);
    }
}

// Zero out all gradients
void zero_gradients(MixerModel* model) {
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t size = vocab * embed * sizeof(float);
    
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, size));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, size));
    
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        size_t size_conv = block->embed_dim * block->kernel_size * sizeof(float);
        size_t size_up = block->embed_dim * block->hidden_dim * sizeof(float);
        size_t size_down = block->hidden_dim * block->embed_dim * sizeof(float);
        size_t size_norm = block->embed_dim * sizeof(float);
        
        CHECK_CUDA(cudaMemset(block->conv_weight_grad, 0, size_conv));
        CHECK_CUDA(cudaMemset(block->channel_up_weight_grad, 0, size_up));
        CHECK_CUDA(cudaMemset(block->channel_down_weight_grad, 0, size_down));
        CHECK_CUDA(cudaMemset(block->rmsnorm1_weight_grad, 0, size_norm));
        CHECK_CUDA(cudaMemset(block->rmsnorm2_weight_grad, 0, size_norm));
    }
}

// Count model parameters
int count_parameters(MixerModel* model) {
    int total_params = 0;
    total_params += model->vocab_size * model->embed_dim;  // Embedding
    total_params += model->vocab_size * model->embed_dim;  // Output projection
    
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        total_params += block->embed_dim * block->kernel_size;  // Convolution weight
        total_params += model->embed_dim * model->hidden_dim;   // Channel up-projection
        total_params += model->hidden_dim * model->embed_dim;   // Channel down-projection
        total_params += model->embed_dim * 2;  // RMSNorm1 and RMSNorm2
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
    fwrite(&model->hidden_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    fwrite(&model->batch_size, sizeof(int), 1, file);
    
    // Allocate temporary host buffers for parameter transfer
    int vocab = model->vocab_size, embed = model->embed_dim;
    float* h_embedding_weight = (float*)malloc(vocab * embed * sizeof(float));
    float* h_out_proj_weight = (float*)malloc(vocab * embed * sizeof(float));
    float* h_rmsnorm_weight = (float*)malloc(embed * sizeof(float));
    
    // Copy embedding weights from device to host and write to file
    CHECK_CUDA(cudaMemcpy(h_embedding_weight, model->embedding_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_embedding_weight, sizeof(float), vocab * embed, file);
    
    // Copy output projection weights from device to host and write to file
    CHECK_CUDA(cudaMemcpy(h_out_proj_weight, model->out_proj_weight, 
               vocab * embed * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_out_proj_weight, sizeof(float), vocab * embed, file);
    
    // For each mixer block, copy weights to host and write
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int embed = block->embed_dim, hidden = block->hidden_dim;
        
        // Write the kernel size and dilation
        fwrite(&block->kernel_size, sizeof(int), 1, file);
        fwrite(&block->dilation, sizeof(int), 1, file);
        
        // Copy convolution weights from device to host and write to file
        int size_conv = embed * block->kernel_size;
        float* h_conv_weight = (float*)malloc(size_conv * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_conv_weight, block->conv_weight, 
                   size_conv * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_conv_weight, sizeof(float), size_conv, file);
        free(h_conv_weight);
        
        // Copy channel up-projection weights from device to host and write to file
        int size_up = embed * hidden;
        float* h_channel_up_weight = (float*)malloc(size_up * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_channel_up_weight, block->channel_up_weight, 
                   size_up * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_channel_up_weight, sizeof(float), size_up, file);
        free(h_channel_up_weight);
        
        // Copy channel down-projection weights from device to host and write to file
        int size_down = hidden * embed;
        float* h_channel_down_weight = (float*)malloc(size_down * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_channel_down_weight, block->channel_down_weight, 
                   size_down * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_channel_down_weight, sizeof(float), size_down, file);
        free(h_channel_down_weight);
        
        // Copy RMSNorm1 weights
        CHECK_CUDA(cudaMemcpy(h_rmsnorm_weight, block->rmsnorm1_weight, 
                   embed * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_rmsnorm_weight, sizeof(float), embed, file);
        
        // Copy RMSNorm2 weights
        CHECK_CUDA(cudaMemcpy(h_rmsnorm_weight, block->rmsnorm2_weight, 
                   embed * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_rmsnorm_weight, sizeof(float), embed, file);
    }
    
    // Free temporary host buffers
    free(h_embedding_weight);
    free(h_out_proj_weight);
    free(h_rmsnorm_weight);
    
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
    int vocab_size, embed_dim, hidden_dim, num_layers, seq_length, batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize a new model with read dimensions
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    
    // Allocate temporary host buffers for parameter transfer
    float* h_embedding_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_out_proj_weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_rmsnorm_weight = (float*)malloc(embed_dim * sizeof(float));
    
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
        MixerBlock* block = model->blocks[i];
        
        // Read kernel size and dilation
        int kernel_size, dilation;
        fread(&kernel_size, sizeof(int), 1, file);
        fread(&dilation, sizeof(int), 1, file);
        block->kernel_size = kernel_size;
        block->dilation = dilation;
        
        // Read convolution weights to host buffer
        int size_conv = embed_dim * kernel_size;
        float* h_conv_weight = (float*)malloc(size_conv * sizeof(float));
        fread(h_conv_weight, sizeof(float), size_conv, file);
        // Copy from host to device
        CHECK_CUDA(cudaMemcpy(block->conv_weight, h_conv_weight, 
                   size_conv * sizeof(float), cudaMemcpyHostToDevice));
        free(h_conv_weight);
        
        // Read channel up-projection weights to host buffer
        int size_up = embed_dim * hidden_dim;
        float* h_channel_up_weight = (float*)malloc(size_up * sizeof(float));
        fread(h_channel_up_weight, sizeof(float), size_up, file);
        // Copy from host to device
        CHECK_CUDA(cudaMemcpy(block->channel_up_weight, h_channel_up_weight, 
                   size_up * sizeof(float), cudaMemcpyHostToDevice));
        free(h_channel_up_weight);
        
        // Read channel down-projection weights to host buffer
        int size_down = hidden_dim * embed_dim;
        float* h_channel_down_weight = (float*)malloc(size_down * sizeof(float));
        fread(h_channel_down_weight, sizeof(float), size_down, file);
        // Copy from host to device
        CHECK_CUDA(cudaMemcpy(block->channel_down_weight, h_channel_down_weight, 
                   size_down * sizeof(float), cudaMemcpyHostToDevice));
        free(h_channel_down_weight);
        
        // Read RMSNorm1 weights
        fread(h_rmsnorm_weight, sizeof(float), embed_dim, file);
        CHECK_CUDA(cudaMemcpy(block->rmsnorm1_weight, h_rmsnorm_weight, 
                   embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        // Read RMSNorm2 weights
        fread(h_rmsnorm_weight, sizeof(float), embed_dim, file);
        CHECK_CUDA(cudaMemcpy(block->rmsnorm2_weight, h_rmsnorm_weight, 
                   embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Free temporary host buffers
    free(h_embedding_weight);
    free(h_out_proj_weight);
    free(h_rmsnorm_weight);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return model;
}

// Generate text from the model with temperature-based sampling
void generate_text(MixerModel* model, const char* corpus, size_t corpus_size, int max_new_tokens, float temperature) {
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
    
    // Generate tokens one by one
    for (int i = 0; i < max_new_tokens; i++) {
        // Forward pass
        mixer_model_forward(model, model->d_input_tokens);
        
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
    int embed_dim = 1024;
    int num_layers = 12;
    int seq_length = 2048;
    int batch_size = 6;
    
    MixerModel* model;
    
    if (argc > 1) {
        // Load existing model
        model = load_model(argv[1]);
    } else {
        // Create new model
        model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    }
    
    printf("Model with %d parameters\n", count_parameters(model));
    
    size_t text_size;
    char* text = load_text_file("combined_corpus.txt", &text_size);
    printf("Loaded text corpus with %zu bytes\n", text_size);
    
    float learning_rate = 0.0001f;
    int total_training_steps = 100000;
    
    printf("Training for %d total steps with learning rate %.6f\n", 
           total_training_steps, learning_rate);
    
    int* h_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    time_t start_time = time(NULL);

    for (int step = 0; step < total_training_steps; step++) {
        // Get random batch from text
        get_random_batch(text, text_size, batch_size, seq_length, h_input_tokens, h_target_tokens);
        
        // Copy input and target tokens to device
        CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_input_tokens, 
                   batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(model->d_target_tokens, h_target_tokens, 
                   batch_size * seq_length * sizeof(int), cudaMemcpyHostToDevice));
        
        // Forward pass
        mixer_model_forward(model, model->d_input_tokens);
        
        // Compute loss and gradients in one step
        float step_loss = compute_loss_and_gradients(model, model->d_target_tokens);
        
        // Backward pass and update
        mixer_model_backward(model);
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
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_mixer_model.bin", localtime(&now));
    save_model(model, model_fname);
    
    free(h_input_tokens);
    free(h_target_tokens);
    free(text);
    free_mixer_model(model);
    return 0;
}