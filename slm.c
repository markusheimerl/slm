#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h> // For FLT_MIN

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
        fprintf(stderr, "cuBLAS error in %s:%d: Error Code %d\n", __FILE__, __LINE__, \
                (int)status); \
        /* Print more specific cuBLAS error messages if needed */ \
        /* switch(status) { case CUBLAS_STATUS_NOT_INITIALIZED: ... } */ \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// MixerBlock structure – all pointers refer to device memory.
typedef struct {
    // --- GQA Parameters ---
    float* q_proj_weight;         // [embed_dim x (num_q_heads * head_dim)]
    float* q_proj_weight_grad;
    float* q_proj_m;
    float* q_proj_v;

    float* k_proj_weight;         // [embed_dim x (num_kv_heads * head_dim)]
    float* k_proj_weight_grad;
    float* k_proj_m;
    float* k_proj_v;

    float* v_proj_weight;         // [embed_dim x (num_kv_heads * head_dim)]
    float* v_proj_weight_grad;
    float* v_proj_m;
    float* v_proj_v;

    float* o_proj_weight;         // [(num_q_heads * head_dim) x embed_dim]
    float* o_proj_weight_grad;
    float* o_proj_m;
    float* o_proj_v;

    // Channel mixing parameters
    float* channel_up_weight;          // [embed_dim x hidden_dim]
    float* channel_up_weight_grad;
    float* channel_up_m;
    float* channel_up_v;

    float* channel_down_weight;        // [hidden_dim x embed_dim]
    float* channel_down_weight_grad;
    float* channel_down_m;
    float* channel_down_v;

    // RMSNorm parameters
    float* rmsnorm1_weight;             // [embed_dim]
    float* rmsnorm1_weight_grad;
    float* rmsnorm1_m;
    float* rmsnorm1_v;

    float* rmsnorm2_weight;             // [embed_dim]
    float* rmsnorm2_weight_grad;
    float* rmsnorm2_m;
    float* rmsnorm2_v;

    // --- Forward pass buffers ---
    float* input_buffer;   // [batch, seq, embed] - Saved input for backward
    float* residual;       // [batch, seq, embed] - Output of first residual add
    float* normalized1;    // [batch, seq, embed] - Output of RMSNorm1
    float* normalized2;    // [batch, seq, embed] - Output of RMSNorm2

    // GQA Buffers
    float* q_proj;         // [batch * seq, num_q_heads * head_dim] - Query projection
    float* k_proj;         // [batch * seq, num_kv_heads * head_dim] - Key projection
    float* v_proj;         // [batch * seq, num_kv_heads * head_dim] - Value projection
    float* q_reshaped;     // [batch, num_q_heads, seq, head_dim]
    float* k_reshaped;     // [batch, num_kv_heads, seq, head_dim]
    float* v_reshaped;     // [batch, num_kv_heads, seq, head_dim]
    float* attn_scores;    // [batch, num_q_heads, seq, seq] - Raw scores (Q @ K^T / sqrt(head_dim)) + mask
    float* attn_softmax;   // [batch, num_q_heads, seq, seq] - Softmax output
    float* attn_output_reshaped; // [batch, num_q_heads, seq, head_dim] - Output of Attention @ V
    float* attn_output;    // [batch, seq, num_q_heads * head_dim] - Reshaped Attention output
    float* gqa_output;     // [batch, seq, embed] - Final GQA output after o_proj

    // Channel Mixing Buffers
    float* channel_up_output;   // [batch, seq, hidden_dim]
    float* channel_up_activated; // [batch, seq, hidden_dim]
    // float* channel_mixed;  // Output of block is stored directly in the next layer's input

    // --- Backward pass buffers ---
    float* d_output_block;       // gradient from next layer [batch, seq, embed] - Saved input grad for backward
    float* d_residual;       // [batch, seq, embed] - grad w.r.t residual buffer

    // GQA Gradient Buffers
    float* d_gqa_output;   // [batch, seq, embed] - grad w.r.t final GQA output
    float* d_attn_output;  // [batch, seq, num_q_heads * head_dim]
    float* d_attn_output_reshaped; // [batch, num_q_heads, seq, head_dim]
    float* d_attn_softmax; // [batch, num_q_heads, seq, seq] - grad w.r.t softmax output (used by softmax_backward)
    float* d_attn_scores;  // [batch, num_q_heads, seq, seq] - grad w.r.t scores (output of softmax_backward)
    float* d_q_reshaped;   // [batch, num_q_heads, seq, head_dim]
    float* d_k_reshaped;   // [batch, num_kv_heads, seq, head_dim] - Accumulates grads for shared heads
    float* d_v_reshaped;   // [batch, num_kv_heads, seq, head_dim] - Accumulates grads for shared heads
    float* d_q_proj;       // [batch * seq, num_q_heads * head_dim]
    float* d_k_proj;       // [batch * seq, num_kv_heads * head_dim]
    float* d_v_proj;       // [batch * seq, num_kv_heads * head_dim]

    // Channel Mixing Gradient Buffers
    // float* d_channel_mixed; // grad w.r.t channel_mixed is d_output_block
    float* d_channel_up_activated; // [batch, seq, hidden_dim]
    float* d_channel_up_output; // [batch, seq, hidden_dim]

    // RMSNorm Gradient Buffers
    float* d_normalized1;   // [batch, seq, embed]
    float* d_normalized2;   // [batch, seq, embed]
    // float* d_input; // Final gradient output of the block backward pass

    // RMSNorm Variance Cache
    float* rms_vars1;           // [batch, seq] for RMSNorm1 variance caching
    float* rms_vars2;           // [batch, seq] for RMSNorm2 variance caching

    // Dimensions:
    int embed_dim;
    int hidden_dim;        // typically 4x embed_dim for MLP
    int seq_length;
    int num_q_heads;       // Number of query heads
    int num_kv_heads;      // Number of key/value heads (must divide num_q_heads)
    int head_dim;          // Dimension of each head
    int q_heads_per_kv;    // num_q_heads / num_kv_heads
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
    int hidden_dim;        // MLP hidden dimension
    int num_layers;
    int seq_length;
    int batch_size;
    int num_q_heads;       // GQA Query heads per layer
    int num_kv_heads;      // GQA Key/Value heads per layer

    // cuBLAS handle
    cublasHandle_t cublas_handle;
} MixerModel;

// --- Activation Kernels ---

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float x = input[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sig;
    }
}

__global__ void silu_deriv_mult_kernel(const float* pre, const float* grad_in, float* grad_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = pre[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        float deriv = sig + x * sig * (1.0f - sig); // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad_out[idx] = grad_in[idx] * deriv;
    }
}

// --- RMSNorm Kernels ---

__global__ void rmsnorm_forward_kernel(const float* input, float* output, float* vars,
                                      const float* weight, int batch_size, int seq_length,
                                      int embed_dim) {
    // Grid-stride loop pattern
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch_size * seq_length;
         idx += gridDim.x * blockDim.x)
    {
        int b = idx / seq_length;
        int s = idx % seq_length;
        int offset = (b * seq_length + s) * embed_dim;

        // Calculate variance
        float sum_squares = 0.0f;
        for (int e = 0; e < embed_dim; e++) {
            float val = input[offset + e];
            sum_squares += val * val;
        }
        float rms = rsqrtf(sum_squares / embed_dim + 1e-5f); // Use rsqrt for efficiency
        vars[idx] = 1.0f / rms; // Store 1/rsqrt = sqrt(var + eps) for backward

        // Normalize and scale
        for (int e = 0; e < embed_dim; e++) {
            output[offset + e] = input[offset + e] * rms * weight[e];
        }
    }
}

__global__ void rmsnorm_backward_kernel(const float* input, const float* grad_out,
                                      const float* vars, const float* weight,
                                      float* grad_in, float* grad_weight,
                                      int batch_size, int seq_length, int embed_dim) {
    // Shared memory for grad_weight reduction
    extern __shared__ float s_grad_weight[]; // Size embed_dim * sizeof(float)

    // Initialize shared memory
    if (threadIdx.x < embed_dim) {
        s_grad_weight[threadIdx.x] = 0.0f;
    }
    __syncthreads(); // Ensure initialization is complete

    // Grid-stride loop pattern
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch_size * seq_length;
         idx += gridDim.x * blockDim.x)
    {
        int b = idx / seq_length;
        int s = idx % seq_length;
        int offset = (b * seq_length + s) * embed_dim;
        float rms_inv = 1.0f / vars[idx]; // vars stores sqrt(var + eps), so this is 1/sqrt(var+eps)

        float sum_g_norm_x = 0.0f; // sum(grad_out * normalized_x)
        float sum_g_w = 0.0f;      // sum(grad_out * weight)

        // Calculate intermediate sums and grad_weight contribution for this sample
        for (int e = 0; e < embed_dim; e++) {
            float norm_x = input[offset + e] * rms_inv; // normalized x
            float g_out = grad_out[offset + e];
            sum_g_norm_x += g_out * norm_x;
            sum_g_w += g_out * weight[e];

            // Accumulate grad_weight in shared memory using atomics (within block)
             atomicAdd(&s_grad_weight[e], g_out * norm_x);
           // grad_weight[e] += g_out * norm_x; // Non-atomic version - requires reduction later
        }

        // Calculate grad_in
        float inv_embed_dim = 1.0f / embed_dim;
        for (int e = 0; e < embed_dim; e++) {
            float norm_x = input[offset + e] * rms_inv; // normalized x
            grad_in[offset + e] = grad_out[offset + e] * weight[e] * rms_inv
                                - norm_x * sum_g_norm_x * inv_embed_dim * rms_inv;
        }
    }

    // Wait for all threads in block to finish sample processing
     __syncthreads();

    // Reduce shared memory grad_weight contributions to global memory
    if (threadIdx.x < embed_dim) {
       atomicAdd(&grad_weight[threadIdx.x], s_grad_weight[threadIdx.x]);
    }
}


// --- Embedding Kernels ---

__global__ void apply_embedding_kernel(const int* input_tokens, const float* embedding_weight, float* embeddings,
                                         int batch_size, int seq_length, int embed_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_length;
    if(idx < total) {
        int token = input_tokens[idx];
        // Basic bounds check for safety
        if(token < 0 || token >= vocab_size) {
             // Handle invalid token, e.g., map to a special token like 0 (if exists) or skip
             token = 0; // Or handle error appropriately
        }
        int input_offset = token * embed_dim;
        int output_offset = idx * embed_dim;
        // Copy embedding vector
        for (int e = 0; e < embed_dim; e++) {
            embeddings[output_offset + e] = embedding_weight[input_offset + e];
        }
    }
}

__global__ void embedding_backward_kernel(const int* input_tokens, const float* d_embeddings,
                                        float* embedding_grad, int batch_size, int seq_length,
                                        int embed_dim, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_length;

    if (idx < total) {
        int token = input_tokens[idx];
        // Bounds check
        if (token >= 0 && token < vocab_size) {
            int grad_offset = token * embed_dim;
            int input_grad_offset = idx * embed_dim;
            // Add gradient contribution to the corresponding embedding vector atomically
            for (int e = 0; e < embed_dim; e++) {
                atomicAdd(&embedding_grad[grad_offset + e], d_embeddings[input_grad_offset + e]);
            }
        }
    }
}

// --- Loss Kernels ---

__global__ void softmax_cross_entropy_kernel(const float* logits, const int* targets,
                                           float* d_logits, float* loss_buffer,
                                           int batch_size, int seq_length, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_length;
    if (idx < total) {
        int target = targets[idx];
        int logits_offset = idx * vocab_size;

        // Find max logit for numerical stability
        float max_val = -FLT_MAX; // Use float.h max
        for (int v = 0; v < vocab_size; v++) {
             max_val = fmaxf(max_val, logits[logits_offset + v]);
           // max_val = (logits[logits_offset + v] > max_val) ? logits[logits_offset + v] : max_val;
        }

        // Compute exp(logits - max) and sum
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float exp_val = expf(logits[logits_offset + v] - max_val);
            d_logits[logits_offset + v] = exp_val; // Store exp values temporarily
            sum_exp += exp_val;
        }

        float inv_sum_exp = 1.0f / (sum_exp + 1e-10f); // Add epsilon for safety

        // Compute probability, gradient (prob - target), and loss
        float loss = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float prob = d_logits[logits_offset + v] * inv_sum_exp;
            d_logits[logits_offset + v] = prob; // Store probability
            if (v == target) {
                d_logits[logits_offset + v] -= 1.0f; // Calculate gradient (prob - 1 for target class)
                loss = -logf(prob + 1e-10f); // Calculate cross-entropy loss term
            }
            // Gradient for non-target classes is just prob
        }

        // Store loss for this position (will be averaged later)
        loss_buffer[idx] = loss;
    }
}

// --- AdamW Optimizer Kernel ---

__global__ void adamw_update_kernel(float* weight, const float* grad, float* m, float* v,
                                   float beta1, float beta2, float epsilon, float learning_rate,
                                   float weight_decay, float alpha_t, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        float g = grad[idx] / scale; // Scale gradient by batch size * seq length

        // AdamW Update Steps
        // Apply weight decay BEFORE momentum update for true AdamW
        float wd_adjusted_weight = weight[idx] * (1.0f - learning_rate * weight_decay);

        // Momentum updates
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        // Bias corrected update
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);

        // Final weight update
        weight[idx] = wd_adjusted_weight - update;
        // weight[idx] -= update; // Adam update (no weight decay applied here)
    }
}


// --- Grouped Query Attention Kernels ---

// Reshape [batch * seq, num_heads * head_dim] to [batch, num_heads, seq, head_dim]
// Mode 0: Q, Mode 1: K, Mode 2: V
__global__ void reshape_qkv_kernel(const float* input, float* output,
                                  int batch_size, int seq_length, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * num_heads * head_dim;

    if (idx < total_elements) {
        // Calculate source indices (input layout: [batch * seq, num_heads * head_dim])
        int total_head_dim = num_heads * head_dim;
        int bs_idx = idx / total_head_dim; // Index in batch*seq dimension
        int head_all_dim_idx = idx % total_head_dim; // Index within the combined head dimension

        int b = bs_idx / seq_length;
        int s = bs_idx % seq_length;
        int h = head_all_dim_idx / head_dim;
        int d = head_all_dim_idx % head_dim;

        // Calculate destination index (output layout: [batch, num_heads, seq, head_dim])
        int dest_idx = b * (num_heads * seq_length * head_dim) +
                       h * (seq_length * head_dim) +
                       s * head_dim +
                       d;

        output[dest_idx] = input[idx];
    }
}

// Reshape [batch, num_heads, seq, head_dim] back to [batch * seq, num_heads * head_dim]
__global__ void reshape_output_kernel(const float* input, float* output,
                                     int batch_size, int seq_length, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * num_heads * head_dim;

    if (idx < total_elements) {
        // Calculate source indices (input layout: [batch, num_heads, seq, head_dim])
        int b = idx / (num_heads * seq_length * head_dim);
        int rem1 = idx % (num_heads * seq_length * head_dim);
        int h = rem1 / (seq_length * head_dim);
        int rem2 = rem1 % (seq_length * head_dim);
        int s = rem2 / head_dim;
        int d = rem2 % head_dim;

        // Calculate destination index (output layout: [batch * seq, num_heads * head_dim])
        int dest_bs_idx = b * seq_length + s;
        int dest_head_all_dim_idx = h * head_dim + d;
        int dest_idx = dest_bs_idx * (num_heads * head_dim) + dest_head_all_dim_idx;

        output[dest_idx] = input[idx];
    }
}


// Calculates Attention Scores: scores = Q @ K^T / sqrt(head_dim)
// Q: [batch, num_q_heads, seq, head_dim]
// K: [batch, num_kv_heads, seq, head_dim]
// Output: [batch, num_q_heads, seq, seq]
__global__ void attention_scores_kernel(const float* q, const float* k, float* scores,
                                       int batch_size, int seq_length,
                                       int num_q_heads, int num_kv_heads, int head_dim,
                                       int q_heads_per_kv) {
    int b = blockIdx.z;
    int h_q = blockIdx.y; // Query head index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Target sequence position (row)
    int j = threadIdx.y; // Source sequence position (col)

    if (b >= batch_size || h_q >= num_q_heads || i >= seq_length || j >= seq_length) return;

    // Determine the corresponding KV head
    int h_kv = h_q / q_heads_per_kv;

    // Calculate indices for Q and K
    int q_offset = b * num_q_heads * seq_length * head_dim + h_q * seq_length * head_dim + i * head_dim;
    int k_offset = b * num_kv_heads * seq_length * head_dim + h_kv * seq_length * head_dim + j * head_dim;

    // Calculate dot product
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        score += q[q_offset + d] * k[k_offset + d];
    }

    // Scale
    score *= rsqrtf((float)head_dim);

    // Write to output scores matrix
    int score_idx = b * num_q_heads * seq_length * seq_length + h_q * seq_length * seq_length + i * seq_length + j;
    scores[score_idx] = score;
}


// Applies causal mask: sets elements where j > i to -infinity (or large negative)
// scores: [batch, num_q_heads, seq, seq]
__global__ void apply_causal_mask_kernel(float* scores, int batch_size, int num_q_heads, int seq_length) {
    int b = blockIdx.z;
    int h = blockIdx.y; // Head index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Target sequence position (row)
    int j = threadIdx.y; // Source sequence position (col)

    if (b >= batch_size || h >= num_q_heads || i >= seq_length || j >= seq_length) return;

    // Apply mask
    if (j > i) {
        int score_idx = b * num_q_heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length + j;
        scores[score_idx] = -1e9f; // Large negative number instead of -inf for stability
    }
}

// Softmax along the last dimension (key sequence length)
// input/output: [batch, num_q_heads, seq, seq]
__global__ void softmax_forward_kernel(float* data, int batch_size, int num_q_heads, int seq_length) {
    // Each thread block handles one row of the attention matrix for one head/batch
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x; // Target sequence position (row index)

    if (b >= batch_size || h >= num_q_heads || i >= seq_length) return;

    int offset = b * num_q_heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length;
    float* row_data = data + offset;

    // Shared memory for reduction
    extern __shared__ float sdata[]; // Size should be at least seq_length

    // --- Find Max ---
    float max_val = -FLT_MAX;
    // Load data into shared memory and find max in parallel
    for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
        sdata[j] = row_data[j];
        max_val = fmaxf(max_val, sdata[j]);
    }
     __syncthreads(); // Ensure all data is loaded

    // Parallel reduction for max
    // (Can optimize further with warp-level reductions)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
       if (threadIdx.x < s) {
          float other_max = (threadIdx.x + s < seq_length) ? sdata[threadIdx.x + s] : -FLT_MAX; // Avoid reading out of bounds if seq_length not power of 2
          max_val = fmaxf(max_val, other_max); // Update local max
          sdata[threadIdx.x] = max_val; // Store intermediate max in shared mem for reduction
       }
       __syncthreads();
    }
     if (threadIdx.x == 0) {
        sdata[0] = max_val; // Final max in sdata[0]
    }
    __syncthreads(); // Ensure final max is visible
    max_val = sdata[0]; // All threads read the final max

    // --- Calculate Exp Sum ---
    float sum_exp = 0.0f;
    // Subtract max, compute exp, and sum in parallel
    for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
        sdata[j] = expf(sdata[j] - max_val);
        sum_exp += sdata[j];
    }
     __syncthreads();

    // Parallel reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
       if (threadIdx.x < s) {
          float other_sum = (threadIdx.x + s < seq_length) ? sdata[threadIdx.x + s] : 0.0f;
          sum_exp += other_sum;
           sdata[threadIdx.x] = sum_exp; // Store intermediate sum
       }
       __syncthreads();
    }
    if (threadIdx.x == 0) {
        sdata[0] = sum_exp; // Final sum in sdata[0]
    }
     __syncthreads();
     sum_exp = sdata[0] + 1e-10f; // All threads read final sum (add epsilon)

    // --- Normalize ---
    // Divide by sum
    for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
        row_data[j] = sdata[j] / sum_exp; // Write back normalized value
    }
}


// Calculates Weighted Values: output = attn_softmax @ V
// attn_softmax: [batch, num_q_heads, seq, seq]
// V: [batch, num_kv_heads, seq, head_dim]
// Output: [batch, num_q_heads, seq, head_dim]
__global__ void attention_values_kernel(const float* attn_softmax, const float* v, float* output,
                                       int batch_size, int seq_length,
                                       int num_q_heads, int num_kv_heads, int head_dim,
                                       int q_heads_per_kv) {
    int b = blockIdx.z;
    int h_q = blockIdx.y; // Query head index
    int i = blockIdx.x;   // Target sequence position
    int d = threadIdx.x;  // Head dimension index

    if (b >= batch_size || h_q >= num_q_heads || i >= seq_length || d >= head_dim) return;

    // Determine the corresponding KV head
    int h_kv = h_q / q_heads_per_kv;

    // Calculate offsets
    int softmax_offset_base = b * num_q_heads * seq_length * seq_length + h_q * seq_length * seq_length + i * seq_length;
    int v_offset_base = b * num_kv_heads * seq_length * head_dim + h_kv * seq_length * head_dim;
    int output_offset = b * num_q_heads * seq_length * head_dim + h_q * seq_length * head_dim + i * head_dim + d;

    float weighted_value = 0.0f;
    for (int j = 0; j < seq_length; ++j) { // Iterate over source sequence positions
        float softmax_prob = attn_softmax[softmax_offset_base + j];
        float value = v[v_offset_base + j * head_dim + d];
        weighted_value += softmax_prob * value;
    }

    output[output_offset] = weighted_value;
}


// --- GQA Backward Kernels ---

// Backward reshape: [batch * seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
__global__ void reshape_output_backward_kernel(const float* grad_output, float* grad_input,
                                             int batch_size, int seq_length, int num_heads, int head_dim) {
    // This kernel structure is identical to reshape_output_kernel, just swapping input/output roles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_length * num_heads * head_dim;

    if (idx < total_elements) {
        // Calculate grad_input indices (layout: [batch, num_heads, seq, head_dim])
        int b = idx / (num_heads * seq_length * head_dim);
        int rem1 = idx % (num_heads * seq_length * head_dim);
        int h = rem1 / (seq_length * head_dim);
        int rem2 = rem1 % (seq_length * head_dim);
        int s = rem2 / head_dim;
        int d = rem2 % head_dim;
        int input_idx = idx; // Direct mapping

        // Calculate grad_output index (layout: [batch * seq, num_heads * head_dim])
        int out_bs_idx = b * seq_length + s;
        int out_head_all_dim_idx = h * head_dim + d;
        int output_idx = out_bs_idx * (num_heads * head_dim) + out_head_all_dim_idx;

        grad_input[input_idx] = grad_output[output_idx];
    }
}


// Backward pass for attention values: Computes grad_attn_softmax and grad_v_reshaped
// grad_output: [batch, num_q_heads, seq, head_dim] (gradient w.r.t attention_values output)
// attn_softmax: [batch, num_q_heads, seq, seq] (from forward)
// V_reshaped: [batch, num_kv_heads, seq, head_dim] (from forward)
// grad_attn_softmax: [batch, num_q_heads, seq, seq] (output)
// grad_v_reshaped: [batch, num_kv_heads, seq, head_dim] (output, accumulated)
__global__ void attention_values_backward_kernel(const float* grad_output, const float* attn_softmax, const float* v_reshaped,
                                                float* grad_attn_softmax, float* grad_v_reshaped,
                                                int batch_size, int seq_length,
                                                int num_q_heads, int num_kv_heads, int head_dim,
                                                int q_heads_per_kv) {
    int b = blockIdx.z;
    int h_q = blockIdx.y; // Query head index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Target sequence position (row in softmax)
    int j = threadIdx.y; // Source sequence position (col in softmax)

    if (b >= batch_size || h_q >= num_q_heads || i >= seq_length || j >= seq_length) return;

    // Corresponding KV head
    int h_kv = h_q / q_heads_per_kv;

    // Calculate grad_attn_softmax[b, h_q, i, j] = sum_d (grad_output[b, h_q, i, d] * V[b, h_kv, j, d])
    float d_softmax_ij = 0.0f;
    int grad_out_offset = b * num_q_heads * seq_length * head_dim + h_q * seq_length * head_dim + i * head_dim;
    int v_offset = b * num_kv_heads * seq_length * head_dim + h_kv * seq_length * head_dim + j * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        d_softmax_ij += grad_output[grad_out_offset + d] * v_reshaped[v_offset + d];
    }
    int softmax_grad_idx = b * num_q_heads * seq_length * seq_length + h_q * seq_length * seq_length + i * seq_length + j;
    grad_attn_softmax[softmax_grad_idx] = d_softmax_ij;

    // Calculate grad_v_reshaped[b, h_kv, j, d] = sum_i (grad_output[b, h_q, i, d] * attn_softmax[b, h_q, i, j])
    // This requires iterating over 'i' and accumulating atomically for 'j' and 'd'
    // Let's recalculate V gradient contributions using a different thread indexing for V
    // One thread per (b, h_kv, j, d)
    int b_v = blockIdx.z;
    int h_kv_v = blockIdx.y;
    int j_v = blockIdx.x * blockDim.x + threadIdx.x; // source sequence position
    int d_v = threadIdx.y; // head dim

    if (b_v >= batch_size || h_kv_v >= num_kv_heads || j_v >= seq_length || d_v >= head_dim) return;

    float d_v_bkhjd = 0.0f;
    // Iterate over the query heads associated with this kv head
    for (int h_q_idx = 0; h_q_idx < q_heads_per_kv; ++h_q_idx) {
        int current_h_q = h_kv_v * q_heads_per_kv + h_q_idx;
        if (current_h_q >= num_q_heads) continue; // Should not happen if dimensions are correct

        // Iterate over target sequence positions 'i'
        for (int i_v = 0; i_v < seq_length; ++i_v) {
            int grad_out_offset_v = b_v * num_q_heads * seq_length * head_dim + current_h_q * seq_length * head_dim + i_v * head_dim + d_v;
            int softmax_offset_v = b_v * num_q_heads * seq_length * seq_length + current_h_q * seq_length * seq_length + i_v * seq_length + j_v;
            d_v_bkhjd += grad_output[grad_out_offset_v] * attn_softmax[softmax_offset_v];
        }
    }

    // Atomically add the gradient contribution to grad_v_reshaped
    int v_grad_idx = b_v * num_kv_heads * seq_length * head_dim + h_kv_v * seq_length * head_dim + j_v * head_dim + d_v;
    atomicAdd(&grad_v_reshaped[v_grad_idx], d_v_bkhjd);
}


// Backward pass for softmax. Calculates gradient w.r.t softmax input (scores)
// grad_output: gradient w.r.t softmax output [batch, num_q_heads, seq, seq]
// softmax_output: softmax output from forward [batch, num_q_heads, seq, seq]
// grad_input: gradient w.r.t softmax input [batch, num_q_heads, seq, seq] (output)
__global__ void softmax_backward_kernel(const float* grad_output, const float* softmax_output, float* grad_input,
                                       int batch_size, int num_q_heads, int seq_length) {
    // Each thread block handles one row of the attention matrix for one head/batch
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x; // Target sequence position (row index)

    if (b >= batch_size || h >= num_q_heads || i >= seq_length) return;

    int offset = b * num_q_heads * seq_length * seq_length + h * seq_length * seq_length + i * seq_length;
    const float* grad_out_row = grad_output + offset;
    const float* softmax_out_row = softmax_output + offset;
    float* grad_in_row = grad_input + offset;

    // Shared memory for reduction
    extern __shared__ float sdata[]; // Size should be at least seq_length

    // --- Calculate sum(grad_output * softmax_output) ---
    float sum_g_s = 0.0f;
    // Load data into shared memory and compute product sum in parallel
    for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
        sdata[j] = grad_out_row[j] * softmax_out_row[j];
        sum_g_s += sdata[j];
    }
     __syncthreads(); // Ensure all products are computed

    // Parallel reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float other_sum = (threadIdx.x + s < seq_length) ? sdata[threadIdx.x + s] : 0.0f;
            sum_g_s += other_sum;
            sdata[threadIdx.x] = sum_g_s; // Store intermediate sum
        }
        __syncthreads();
    }
     if (threadIdx.x == 0) {
        sdata[0] = sum_g_s; // Final sum in sdata[0]
    }
    __syncthreads(); // Ensure final sum is visible
    sum_g_s = sdata[0]; // All threads read the final sum

    // --- Calculate grad_input ---
    // grad_input = (grad_output - sum_g_s) * softmax_output
    for (int j = threadIdx.x; j < seq_length; j += blockDim.x) {
        // Apply causal mask implicitly: if j > i, softmax_output should be ~0, so gradient is ~0.
        // No explicit masking needed here if softmax_output already incorporates the mask effect.
        grad_in_row[j] = (grad_out_row[j] - sum_g_s) * softmax_out_row[j];

        // Explicitly zero out gradient for masked positions if needed (defensive)
        // if (j > i) {
        //     grad_in_row[j] = 0.0f;
        // }
    }
}


// Backward pass for attention scores: Computes grad_q_reshaped and grad_k_reshaped
// grad_scores: [batch, num_q_heads, seq, seq] (gradient w.r.t attention scores output, after softmax backward)
// Q_reshaped: [batch, num_q_heads, seq, head_dim] (from forward)
// K_reshaped: [batch, num_kv_heads, seq, head_dim] (from forward)
// grad_q_reshaped: [batch, num_q_heads, seq, head_dim] (output)
// grad_k_reshaped: [batch, num_kv_heads, seq, head_dim] (output, accumulated)
__global__ void attention_scores_backward_kernel(const float* grad_scores, const float* q_reshaped, const float* k_reshaped,
                                                 float* grad_q_reshaped, float* grad_k_reshaped,
                                                 int batch_size, int seq_length,
                                                 int num_q_heads, int num_kv_heads, int head_dim,
                                                 int q_heads_per_kv) {
    int b = blockIdx.z;
    int h_q = blockIdx.y; // Query head index
    int i = blockIdx.x;   // Target sequence position (row in scores)
    int d = threadIdx.x;  // Head dimension index

    if (b >= batch_size || h_q >= num_q_heads || i >= seq_length || d >= head_dim) return;

    int h_kv = h_q / q_heads_per_kv;
    float scale = rsqrtf((float)head_dim);

    // Calculate grad_q_reshaped[b, h_q, i, d] = sum_j (grad_scores[b, h_q, i, j] * K[b, h_kv, j, d]) * scale
    float d_q_biqd = 0.0f;
    int grad_scores_offset = b * num_q_heads * seq_length * seq_length + h_q * seq_length * seq_length + i * seq_length;
    int k_offset_base = b * num_kv_heads * seq_length * head_dim + h_kv * seq_length * head_dim;
    for (int j = 0; j < seq_length; ++j) {
        d_q_biqd += grad_scores[grad_scores_offset + j] * k_reshaped[k_offset_base + j * head_dim + d];
    }
    int q_grad_idx = b * num_q_heads * seq_length * head_dim + h_q * seq_length * head_dim + i * head_dim + d;
    grad_q_reshaped[q_grad_idx] = d_q_biqd * scale;


    // Calculate grad_k_reshaped[b, h_kv, j, d] = sum_i (grad_scores[b, h_q, i, j] * Q[b, h_q, i, d]) * scale
    // Again, requires accumulation. Let's index by K dimensions.
    // One thread per (b, h_kv, j, d)
    int b_k = blockIdx.z;
    int h_kv_k = blockIdx.y;
    int j_k = blockIdx.x * blockDim.x + threadIdx.x; // source sequence position
    int d_k = threadIdx.y; // head dim

    if (b_k >= batch_size || h_kv_k >= num_kv_heads || j_k >= seq_length || d_k >= head_dim) return;

    float d_k_bkhjd = 0.0f;
    // Iterate over the query heads associated with this kv head
    for (int h_q_idx = 0; h_q_idx < q_heads_per_kv; ++h_q_idx) {
        int current_h_q = h_kv_k * q_heads_per_kv + h_q_idx;
        if (current_h_q >= num_q_heads) continue;

        // Iterate over target sequence positions 'i'
        for (int i_k = 0; i_k < seq_length; ++i_k) {
             int grad_scores_offset_k = b_k * num_q_heads * seq_length * seq_length + current_h_q * seq_length * seq_length + i_k * seq_length + j_k;
             int q_offset_k = b_k * num_q_heads * seq_length * head_dim + current_h_q * seq_length * head_dim + i_k * head_dim + d_k;
             d_k_bkhjd += grad_scores[grad_scores_offset_k] * q_reshaped[q_offset_k];
        }
    }

    // Atomically add the gradient contribution to grad_k_reshaped
    int k_grad_idx = b_k * num_kv_heads * seq_length * head_dim + h_kv_k * seq_length * head_dim + j_k * head_dim + d_k;
    atomicAdd(&grad_k_reshaped[k_grad_idx], d_k_bkhjd * scale);
}

// Backward reshape: [batch, num_heads, seq, head_dim] -> [batch * seq, num_heads * head_dim]
// Accumulates gradients for K and V
__global__ void reshape_qkv_backward_kernel(const float* grad_q_reshaped, const float* grad_k_reshaped, const float* grad_v_reshaped,
                                          float* grad_q_proj, float* grad_k_proj, float* grad_v_proj,
                                          int batch_size, int seq_length,
                                          int num_q_heads, int num_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements_q = batch_size * seq_length * num_q_heads * head_dim;
    int total_elements_kv = batch_size * seq_length * num_kv_heads * head_dim;

    // --- Process Q gradient ---
    if (idx < total_elements_q) {
        // Calculate source indices (grad_q_reshaped layout: [batch, num_q_heads, seq, head_dim])
        int b = idx / (num_q_heads * seq_length * head_dim);
        int rem1 = idx % (num_q_heads * seq_length * head_dim);
        int h = rem1 / (seq_length * head_dim);
        int rem2 = rem1 % (seq_length * head_dim);
        int s = rem2 / head_dim;
        int d = rem2 % head_dim;
        int src_idx = idx; // Direct mapping

        // Calculate destination index (grad_q_proj layout: [batch * seq, num_q_heads * head_dim])
        int dest_bs_idx = b * seq_length + s;
        int dest_head_all_dim_idx = h * head_dim + d;
        int dest_idx = dest_bs_idx * (num_q_heads * head_dim) + dest_head_all_dim_idx;

        grad_q_proj[dest_idx] = grad_q_reshaped[src_idx];
    }

    // --- Process K gradient ---
    if (idx < total_elements_kv) {
        // Calculate source indices (grad_k_reshaped layout: [batch, num_kv_heads, seq, head_dim])
        int b = idx / (num_kv_heads * seq_length * head_dim);
        int rem1 = idx % (num_kv_heads * seq_length * head_dim);
        int h = rem1 / (seq_length * head_dim);
        int rem2 = rem1 % (seq_length * head_dim);
        int s = rem2 / head_dim;
        int d = rem2 % head_dim;
        int src_idx = idx; // Direct mapping

        // Calculate destination index (grad_k_proj layout: [batch * seq, num_kv_heads * head_dim])
        int dest_bs_idx = b * seq_length + s;
        int dest_head_all_dim_idx = h * head_dim + d;
        int dest_idx = dest_bs_idx * (num_kv_heads * head_dim) + dest_head_all_dim_idx;

        // Assign (no accumulation needed here as grad_k_reshaped was accumulated previously)
        grad_k_proj[dest_idx] = grad_k_reshaped[src_idx];
    }

    // --- Process V gradient ---
    if (idx < total_elements_kv) {
         // Calculate source indices (grad_v_reshaped layout: [batch, num_kv_heads, seq, head_dim])
        int b = idx / (num_kv_heads * seq_length * head_dim);
        int rem1 = idx % (num_kv_heads * seq_length * head_dim);
        int h = rem1 / (seq_length * head_dim);
        int rem2 = rem1 % (seq_length * head_dim);
        int s = rem2 / head_dim;
        int d = rem2 % head_dim;
        int src_idx = idx; // Direct mapping

        // Calculate destination index (grad_v_proj layout: [batch * seq, num_kv_heads * head_dim])
        int dest_bs_idx = b * seq_length + s;
        int dest_head_all_dim_idx = h * head_dim + d;
        int dest_idx = dest_bs_idx * (num_kv_heads * head_dim) + dest_head_all_dim_idx;

        // Assign (no accumulation needed here as grad_v_reshaped was accumulated previously)
        grad_v_proj[dest_idx] = grad_v_reshaped[src_idx];
    }
}


// --- Initialization and Freeing Functions ---

MixerBlock* init_mixer_block(int embed_dim, int hidden_dim_mlp, int seq_length, int batch_size,
                             int num_q_heads, int num_kv_heads) {
    MixerBlock* block = (MixerBlock*)malloc(sizeof(MixerBlock));
    block->embed_dim = embed_dim;
    block->hidden_dim = hidden_dim_mlp; // MLP hidden dim
    block->seq_length = seq_length;
    block->num_q_heads = num_q_heads;
    block->num_kv_heads = num_kv_heads;

    // Validate GQA dimensions
    if (embed_dim % num_q_heads != 0) {
        fprintf(stderr, "Error: embed_dim (%d) must be divisible by num_q_heads (%d)\n", embed_dim, num_q_heads);
        exit(EXIT_FAILURE);
    }
    if (num_q_heads % num_kv_heads != 0) {
        fprintf(stderr, "Error: num_q_heads (%d) must be divisible by num_kv_heads (%d)\n", num_q_heads, num_kv_heads);
        exit(EXIT_FAILURE);
    }
    block->head_dim = embed_dim / num_q_heads;
    block->q_heads_per_kv = num_q_heads / num_kv_heads;

    printf("Initializing MixerBlock: embed=%d, seq=%d, heads_q=%d, heads_kv=%d, head_dim=%d\n",
           embed_dim, seq_length, num_q_heads, num_kv_heads, block->head_dim);

    // --- GQA Weights and Buffers ---
    int q_dim = num_q_heads * block->head_dim; // Should equal embed_dim
    int kv_dim = num_kv_heads * block->head_dim;
    int o_dim = q_dim; // Output projection input dim

    size_t size_q_proj = embed_dim * q_dim * sizeof(float);
    size_t size_kv_proj = embed_dim * kv_dim * sizeof(float);
    size_t size_o_proj = o_dim * embed_dim * sizeof(float); // Note: o_dim = embed_dim

    // Helper for weight initialization
    auto init_weights = [&](float* d_ptr, size_t num_elements, float scale) {
        float* h_ptr = (float*)malloc(num_elements * sizeof(float));
        for (size_t i = 0; i < num_elements; i++) {
            h_ptr[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
        }
        CHECK_CUDA(cudaMalloc(&d_ptr, num_elements * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ptr);
        return d_ptr; // Return allocated device pointer
    };

     // Helper to allocate and zero device memory
    auto alloc_zero = [&](float* d_ptr, size_t num_elements) {
        CHECK_CUDA(cudaMalloc(&d_ptr, num_elements * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_ptr, 0, num_elements * sizeof(float)));
        return d_ptr;
    };

    // Q, K, V projection weights + grads + Adam states
    float scale_proj = 1.0f / sqrtf((float)embed_dim);
    block->q_proj_weight = init_weights(block->q_proj_weight, embed_dim * q_dim, scale_proj);
    block->k_proj_weight = init_weights(block->k_proj_weight, embed_dim * kv_dim, scale_proj);
    block->v_proj_weight = init_weights(block->v_proj_weight, embed_dim * kv_dim, scale_proj);

    block->q_proj_weight_grad = alloc_zero(block->q_proj_weight_grad, embed_dim * q_dim);
    block->k_proj_weight_grad = alloc_zero(block->k_proj_weight_grad, embed_dim * kv_dim);
    block->v_proj_weight_grad = alloc_zero(block->v_proj_weight_grad, embed_dim * kv_dim);

    block->q_proj_m = alloc_zero(block->q_proj_m, embed_dim * q_dim);
    block->k_proj_m = alloc_zero(block->k_proj_m, embed_dim * kv_dim);
    block->v_proj_m = alloc_zero(block->v_proj_m, embed_dim * kv_dim);

    block->q_proj_v = alloc_zero(block->q_proj_v, embed_dim * q_dim);
    block->k_proj_v = alloc_zero(block->k_proj_v, embed_dim * kv_dim);
    block->v_proj_v = alloc_zero(block->v_proj_v, embed_dim * kv_dim);

    // Output projection weights + grads + Adam states
    float scale_o_proj = 1.0f / sqrtf((float)o_dim);
    block->o_proj_weight = init_weights(block->o_proj_weight, o_dim * embed_dim, scale_o_proj);
    block->o_proj_weight_grad = alloc_zero(block->o_proj_weight_grad, o_dim * embed_dim);
    block->o_proj_m = alloc_zero(block->o_proj_m, o_dim * embed_dim);
    block->o_proj_v = alloc_zero(block->o_proj_v, o_dim * embed_dim);

    // --- Channel Mixing Weights ---
    size_t size_up_channel = embed_dim * block->hidden_dim * sizeof(float);
    float scale_up_channel = 1.0f / sqrtf((float)embed_dim);
    block->channel_up_weight = init_weights(block->channel_up_weight, embed_dim * block->hidden_dim, scale_up_channel);
    block->channel_up_weight_grad = alloc_zero(block->channel_up_weight_grad, embed_dim * block->hidden_dim);
    block->channel_up_m = alloc_zero(block->channel_up_m, embed_dim * block->hidden_dim);
    block->channel_up_v = alloc_zero(block->channel_up_v, embed_dim * block->hidden_dim);

    size_t size_down_channel = block->hidden_dim * embed_dim * sizeof(float);
    float scale_down_channel = 1.0f / sqrtf((float)block->hidden_dim);
     block->channel_down_weight = init_weights(block->channel_down_weight, block->hidden_dim * embed_dim, scale_down_channel);
    block->channel_down_weight_grad = alloc_zero(block->channel_down_weight_grad, block->hidden_dim * embed_dim);
    block->channel_down_m = alloc_zero(block->channel_down_m, block->hidden_dim * embed_dim);
    block->channel_down_v = alloc_zero(block->channel_down_v, block->hidden_dim * embed_dim);

    // --- RMSNorm Weights ---
    size_t rmsnorm_size = embed_dim * sizeof(float);
    float* h_rmsnorm_weight = (float*)malloc(rmsnorm_size);
    for (int i = 0; i < embed_dim; i++) h_rmsnorm_weight[i] = 1.0f;

    // RMSNorm1
    CHECK_CUDA(cudaMalloc(&block->rmsnorm1_weight, rmsnorm_size));
    CHECK_CUDA(cudaMemcpy(block->rmsnorm1_weight, h_rmsnorm_weight, rmsnorm_size, cudaMemcpyHostToDevice));
    block->rmsnorm1_weight_grad = alloc_zero(block->rmsnorm1_weight_grad, embed_dim);
    block->rmsnorm1_m = alloc_zero(block->rmsnorm1_m, embed_dim);
    block->rmsnorm1_v = alloc_zero(block->rmsnorm1_v, embed_dim);

    // RMSNorm2
    CHECK_CUDA(cudaMalloc(&block->rmsnorm2_weight, rmsnorm_size));
    CHECK_CUDA(cudaMemcpy(block->rmsnorm2_weight, h_rmsnorm_weight, rmsnorm_size, cudaMemcpyHostToDevice));
    block->rmsnorm2_weight_grad = alloc_zero(block->rmsnorm2_weight_grad, embed_dim);
    block->rmsnorm2_m = alloc_zero(block->rmsnorm2_m, embed_dim);
    block->rmsnorm2_v = alloc_zero(block->rmsnorm2_v, embed_dim);
    free(h_rmsnorm_weight);


    // --- Allocate Forward Pass Buffers ---
    size_t bse_elements = (size_t)batch_size * seq_length * embed_dim;
    size_t bsh_elements = (size_t)batch_size * seq_length * block->hidden_dim; // MLP hidden

    // Common buffers
    block->input_buffer = alloc_zero(block->input_buffer, bse_elements);
    block->residual = alloc_zero(block->residual, bse_elements);
    block->normalized1 = alloc_zero(block->normalized1, bse_elements);
    block->normalized2 = alloc_zero(block->normalized2, bse_elements);

    // GQA forward buffers
    size_t bs_q_elements = (size_t)batch_size * seq_length * q_dim;
    size_t bs_kv_elements = (size_t)batch_size * seq_length * kv_dim;
    size_t b_nq_s_hd_elements = (size_t)batch_size * num_q_heads * seq_length * block->head_dim; // = bse_elements
    size_t b_nkv_s_hd_elements = (size_t)batch_size * num_kv_heads * seq_length * block->head_dim;
    size_t attn_score_elements = (size_t)batch_size * num_q_heads * seq_length * seq_length;
    size_t bs_o_elements = (size_t)batch_size * seq_length * o_dim; // = bse_elements

    block->q_proj = alloc_zero(block->q_proj, bs_q_elements);
    block->k_proj = alloc_zero(block->k_proj, bs_kv_elements);
    block->v_proj = alloc_zero(block->v_proj, bs_kv_elements);
    block->q_reshaped = alloc_zero(block->q_reshaped, b_nq_s_hd_elements);
    block->k_reshaped = alloc_zero(block->k_reshaped, b_nkv_s_hd_elements);
    block->v_reshaped = alloc_zero(block->v_reshaped, b_nkv_s_hd_elements);
    block->attn_scores = alloc_zero(block->attn_scores, attn_score_elements);
    block->attn_softmax = alloc_zero(block->attn_softmax, attn_score_elements);
    block->attn_output_reshaped = alloc_zero(block->attn_output_reshaped, b_nq_s_hd_elements);
    block->attn_output = alloc_zero(block->attn_output, bs_o_elements);
    block->gqa_output = alloc_zero(block->gqa_output, bse_elements);

    // Channel Mixing forward buffers
    block->channel_up_output = alloc_zero(block->channel_up_output, bsh_elements);
    block->channel_up_activated = alloc_zero(block->channel_up_activated, bsh_elements);

    // RMSNorm variance buffers
    size_t rms_var_elements = (size_t)batch_size * seq_length;
    block->rms_vars1 = alloc_zero(block->rms_vars1, rms_var_elements);
    block->rms_vars2 = alloc_zero(block->rms_vars2, rms_var_elements);


    // --- Allocate Backward Pass Buffers ---
    block->d_output_block = alloc_zero(block->d_output_block, bse_elements);
    block->d_residual = alloc_zero(block->d_residual, bse_elements);

    // GQA backward buffers
    block->d_gqa_output = alloc_zero(block->d_gqa_output, bse_elements);
    block->d_attn_output = alloc_zero(block->d_attn_output, bs_o_elements);
    block->d_attn_output_reshaped = alloc_zero(block->d_attn_output_reshaped, b_nq_s_hd_elements);
    block->d_attn_softmax = alloc_zero(block->d_attn_softmax, attn_score_elements); // Temp buffer for softmax backward
    block->d_attn_scores = alloc_zero(block->d_attn_scores, attn_score_elements);
    block->d_q_reshaped = alloc_zero(block->d_q_reshaped, b_nq_s_hd_elements);
    block->d_k_reshaped = alloc_zero(block->d_k_reshaped, b_nkv_s_hd_elements);
    block->d_v_reshaped = alloc_zero(block->d_v_reshaped, b_nkv_s_hd_elements);
    block->d_q_proj = alloc_zero(block->d_q_proj, bs_q_elements);
    block->d_k_proj = alloc_zero(block->d_k_proj, bs_kv_elements);
    block->d_v_proj = alloc_zero(block->d_v_proj, bs_kv_elements);

    // Channel Mixing backward buffers
    block->d_channel_up_activated = alloc_zero(block->d_channel_up_activated, bsh_elements);
    block->d_channel_up_output = alloc_zero(block->d_channel_up_output, bsh_elements);

    // RMSNorm backward buffers
    block->d_normalized1 = alloc_zero(block->d_normalized1, bse_elements);
    block->d_normalized2 = alloc_zero(block->d_normalized2, bse_elements);
    // d_input is the final output gradient buffer passed into the backward function

    return block;
}

void free_mixer_block(MixerBlock* block) {
    // Free GQA parameters
    CHECK_CUDA(cudaFree(block->q_proj_weight));
    CHECK_CUDA(cudaFree(block->q_proj_weight_grad));
    CHECK_CUDA(cudaFree(block->q_proj_m));
    CHECK_CUDA(cudaFree(block->q_proj_v));
    CHECK_CUDA(cudaFree(block->k_proj_weight));
    CHECK_CUDA(cudaFree(block->k_proj_weight_grad));
    CHECK_CUDA(cudaFree(block->k_proj_m));
    CHECK_CUDA(cudaFree(block->k_proj_v));
    CHECK_CUDA(cudaFree(block->v_proj_weight));
    CHECK_CUDA(cudaFree(block->v_proj_weight_grad));
    CHECK_CUDA(cudaFree(block->v_proj_m));
    CHECK_CUDA(cudaFree(block->v_proj_v));
    CHECK_CUDA(cudaFree(block->o_proj_weight));
    CHECK_CUDA(cudaFree(block->o_proj_weight_grad));
    CHECK_CUDA(cudaFree(block->o_proj_m));
    CHECK_CUDA(cudaFree(block->o_proj_v));

    // Free channel mixing parameters
    CHECK_CUDA(cudaFree(block->channel_up_weight));
    CHECK_CUDA(cudaFree(block->channel_up_weight_grad));
    CHECK_CUDA(cudaFree(block->channel_up_m));
    CHECK_CUDA(cudaFree(block->channel_up_v));
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
    CHECK_CUDA(cudaFree(block->q_proj));
    CHECK_CUDA(cudaFree(block->k_proj));
    CHECK_CUDA(cudaFree(block->v_proj));
    CHECK_CUDA(cudaFree(block->q_reshaped));
    CHECK_CUDA(cudaFree(block->k_reshaped));
    CHECK_CUDA(cudaFree(block->v_reshaped));
    CHECK_CUDA(cudaFree(block->attn_scores));
    CHECK_CUDA(cudaFree(block->attn_softmax));
    CHECK_CUDA(cudaFree(block->attn_output_reshaped));
    CHECK_CUDA(cudaFree(block->attn_output));
    CHECK_CUDA(cudaFree(block->gqa_output));
    CHECK_CUDA(cudaFree(block->channel_up_output));
    CHECK_CUDA(cudaFree(block->channel_up_activated));
    CHECK_CUDA(cudaFree(block->rms_vars1));
    CHECK_CUDA(cudaFree(block->rms_vars2));

    // Free backward pass buffers
    CHECK_CUDA(cudaFree(block->d_output_block));
    CHECK_CUDA(cudaFree(block->d_residual));
    CHECK_CUDA(cudaFree(block->d_gqa_output));
    CHECK_CUDA(cudaFree(block->d_attn_output));
    CHECK_CUDA(cudaFree(block->d_attn_output_reshaped));
    CHECK_CUDA(cudaFree(block->d_attn_softmax));
    CHECK_CUDA(cudaFree(block->d_attn_scores));
    CHECK_CUDA(cudaFree(block->d_q_reshaped));
    CHECK_CUDA(cudaFree(block->d_k_reshaped));
    CHECK_CUDA(cudaFree(block->d_v_reshaped));
    CHECK_CUDA(cudaFree(block->d_q_proj));
    CHECK_CUDA(cudaFree(block->d_k_proj));
    CHECK_CUDA(cudaFree(block->d_v_proj));
    CHECK_CUDA(cudaFree(block->d_channel_up_activated));
    CHECK_CUDA(cudaFree(block->d_channel_up_output));
    CHECK_CUDA(cudaFree(block->d_normalized1));
    CHECK_CUDA(cudaFree(block->d_normalized2));

    free(block);
}

MixerModel* init_mixer_model(int vocab_size, int embed_dim, int num_layers, int seq_length, int batch_size,
                             int num_q_heads, int num_kv_heads) {
    MixerModel* model = (MixerModel*)malloc(sizeof(MixerModel));

    // Store dimensions
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->hidden_dim = embed_dim * 4;  // MLP hidden dim
    model->num_layers = num_layers;
    model->seq_length = seq_length;
    model->batch_size = batch_size;
    model->num_q_heads = num_q_heads;
    model->num_kv_heads = num_kv_heads;

    // Initialize Adam optimizer parameters
    model->beta1 = 0.9f;
    model->beta2 = 0.999f; // AdamW default is often 0.999
    model->epsilon = 1e-8f;
    model->weight_decay = 0.01f; // Common default for AdamW
    model->t = 0;

    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&model->cublas_handle));
    // Enable Tensor Core usage if available and appropriate (requires Volta+ GPUs)
    // CHECK_CUBLAS(cublasSetMathMode(model->cublas_handle, CUBLAS_TENSOR_OP_MATH));

     // Helper for weight initialization
    auto init_weights = [&](float* d_ptr, size_t num_elements, float scale) {
        float* h_ptr = (float*)malloc(num_elements * sizeof(float));
        for (size_t i = 0; i < num_elements; i++) {
            h_ptr[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
        }
        CHECK_CUDA(cudaMalloc(&d_ptr, num_elements * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ptr);
        return d_ptr; // Return allocated device pointer
    };
     // Helper to allocate and zero device memory
    auto alloc_zero = [&](float* d_ptr, size_t num_elements) {
        CHECK_CUDA(cudaMalloc(&d_ptr, num_elements * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_ptr, 0, num_elements * sizeof(float)));
        return d_ptr;
    };


    // --- Embedding parameters ---
    size_t embed_matrix_elements = (size_t)vocab_size * embed_dim;
    float scale_embed = 1.0f / sqrtf((float)embed_dim); // Xavier/Glorot scaling
    model->embedding_weight = init_weights(model->embedding_weight, embed_matrix_elements, scale_embed);
    model->embedding_weight_grad = alloc_zero(model->embedding_weight_grad, embed_matrix_elements);
    model->embedding_m = alloc_zero(model->embedding_m, embed_matrix_elements);
    model->embedding_v = alloc_zero(model->embedding_v, embed_matrix_elements);

    // --- Output projection parameters ---
    model->out_proj_weight = init_weights(model->out_proj_weight, embed_matrix_elements, scale_embed);
    model->out_proj_weight_grad = alloc_zero(model->out_proj_weight_grad, embed_matrix_elements);
    model->out_proj_m = alloc_zero(model->out_proj_m, embed_matrix_elements);
    model->out_proj_v = alloc_zero(model->out_proj_v, embed_matrix_elements);

    // --- Initialize MixerBlocks ---
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++) {
        model->blocks[i] = init_mixer_block(embed_dim, model->hidden_dim, seq_length, batch_size, num_q_heads, num_kv_heads);
    }

    // --- Allocate Forward Pass Buffers ---
    size_t tensor_elements = (size_t)batch_size * seq_length * embed_dim;
    model->embeddings = alloc_zero(model->embeddings, tensor_elements);
    // Allocate space for input + all layer outputs concatenated
    model->block_outputs = alloc_zero(model->block_outputs, (num_layers + 1) * tensor_elements);

    // Logits buffer
    size_t logits_elements = (size_t)batch_size * seq_length * vocab_size;
    model->logits = alloc_zero(model->logits, logits_elements);

    // --- Allocate Backward Pass Buffers ---
    model->d_logits = alloc_zero(model->d_logits, logits_elements);
    model->d_block_outputs = alloc_zero(model->d_block_outputs, (num_layers + 1) * tensor_elements);

    // --- Allocate Persistent Device Buffers for Training ---
    size_t token_elements = (size_t)batch_size * seq_length;
    CHECK_CUDA(cudaMalloc(&model->d_input_tokens, token_elements * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&model->d_target_tokens, token_elements * sizeof(int)));
    model->d_loss_buffer = alloc_zero(model->d_loss_buffer, token_elements);

    // Allocate persistent host buffer for loss computation
    model->h_loss_buffer = (float*)malloc(token_elements * sizeof(float));
    if (!model->h_loss_buffer) {
         fprintf(stderr, "Failed to allocate host loss buffer\n");
         exit(EXIT_FAILURE);
     }


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


// --- Forward and Backward Passes ---

// Forward pass through one MixerBlock with GQA
void mixer_block_forward(MixerBlock* block, float* input, float* output, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int hidden_mlp = block->hidden_dim;
    int num_q_heads = block->num_q_heads;
    int num_kv_heads = block->num_kv_heads;
    int head_dim = block->head_dim;
    int q_heads_per_kv = block->q_heads_per_kv;

    size_t total_bse = (size_t)batch_size * seq * embed;
    size_t total_bsh = (size_t)batch_size * seq * hidden_mlp;
    size_t total_bs = (size_t)batch_size * seq;

    float alpha = 1.0f, beta = 0.0f;
    int threads_per_block = 256; // General purpose threads

    // Save input for backward pass's residual connection
    CHECK_CUDA(cudaMemcpy(block->input_buffer, input, total_bse * sizeof(float), cudaMemcpyDeviceToDevice));

    // --- RMSNorm 1 ---
    int rms_blocks = (batch_size * seq + threads_per_block - 1) / threads_per_block;
    rmsnorm_forward_kernel<<<rms_blocks, threads_per_block>>>(input, block->normalized1, block->rms_vars1,
                                                block->rmsnorm1_weight, batch_size, seq, embed);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // --- Grouped Query Attention ---
    int combined_batch_seq = batch_size * seq;
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int o_dim = q_dim; // Output projection input dim

    // 1. Q, K, V Projections (GEMM: output = input @ W^T)
    // Input: normalized1 [bs, embed] -> Output: q_proj [bs, q_dim] etc.
    // cublasSgemm: C = alpha * op(A) * op(B) + beta * C
    // Here: C=q_proj, A=q_proj_weight, B=normalized1
    // op(A)=A^T (q_dim x embed), op(B)=B (embed x bs) -> C (q_dim x bs)
    // We need C to be [bs, q_dim], so let C=q_proj^T, A=normalized1^T, B=q_proj_weight^T
    // C^T(bs x q_dim) = B^T(bs x embed) @ A^T(embed x q_dim)
    // Let's use standard layout: C(bs x q_dim) = B(bs x embed) @ A(embed x q_dim)
    // C = q_proj (M=bs, N=q_dim)
    // B = normalized1 (M=bs, K=embed)
    // A = q_proj_weight (K=embed, N=q_dim) -> Use A = q_proj_weight (transpose it)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             q_dim, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->q_proj_weight, q_dim,    // A (N x K = q_dim x embed) - Assuming weights stored column-major or need transpose
                             block->normalized1, embed,      // B (K x M = embed x bs)
                             &beta,
                             block->q_proj, q_dim));          // C (N x M = q_dim x bs) - Result needs transpose conceptually if thinking row-major

    // Need to adjust cublas call if weights are row-major (C style)
    // If weights W are [rows, cols] in C-style (row-major):
    // W is stored as W_rowmajor = [row0, row1, ...]
    // cublas expects column-major. W_colmajor = [col0, col1, ...]
    // To compute X @ W (where X=[bs, embed], W=[embed, q_dim]) -> output=[bs, q_dim]
    // C = X @ W -> M=bs, N=q_dim, K=embed
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, W_colmajor, N, X_colmajor, K, beta, C_colmajor, N)
    // If X is row-major input (block->normalized1) and W is row-major (block->q_proj_weight)
    // We want C_rowmajor (block->q_proj) = X_rowmajor @ W_rowmajor
    // This is equivalent to C_colmajor^T = W_colmajor^T @ X_colmajor^T
    // Let C'=C_colmajor^T, W'=W_colmajor^T, X'=X_colmajor^T
    // We can compute C = W @ X using cublas if we treat rows as columns.
    // Let's compute C^T = W^T @ X^T (standard BLAS notation, column major assumed by cublas)
    // C^T needs dimensions [q_dim, bs]
    // W^T needs dimensions [q_dim, embed] (use W with CUBLAS_OP_T, if W is col-major [embed, q_dim])
    // X^T needs dimensions [embed, bs] (use X with CUBLAS_OP_T, if X is col-major [bs, embed])
    // If X and W are row-major:
    // X_rowmajor [bs, embed] -> cublas treats as X_colmajor [embed, bs] (lda=bs)
    // W_rowmajor [embed, q_dim] -> cublas treats as W_colmajor [q_dim, embed] (lda=embed)
    // To get C_rowmajor [bs, q_dim] = X_rowmajor @ W_rowmajor:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, q_dim, bs, embed, alpha, W_colmajor(lda=q_dim), X_colmajor(lda=embed), beta, C_colmajor(lda=q_dim))
    // It seems the first attempt was correct if we assume cuBLAS handles the layout implicitly or weights were stored transposed.
    // Let's re-verify typical cuBLAS usage for C = A*B (A=[M,K], B=[K,N], C=[M,N])
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
    // Here C=q_proj (bs x q_dim), A=normalized1 (bs x embed), B=q_proj_weight (embed x q_dim)
    // M=bs, N=q_dim, K=embed
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             q_dim, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->q_proj_weight, q_dim,    // B (lda=N)
                             block->normalized1, embed,      // A (lda=K)
                             &beta,
                             block->q_proj, q_dim));          // C (lda=N) -> Result [bs, q_dim] (row-major interpretation)

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             kv_dim, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->k_proj_weight, kv_dim,   // B
                             block->normalized1, embed,     // A
                             &beta,
                             block->k_proj, kv_dim));         // C

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             kv_dim, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->v_proj_weight, kv_dim,   // B
                             block->normalized1, embed,     // A
                             &beta,
                             block->v_proj, kv_dim));         // C

    // 2. Reshape Q, K, V
    // [bs, num_heads * head_dim] -> [b, num_heads, s, head_dim]
    size_t qkv_elements = total_bse; // q uses full embed dim
    int reshape_q_blocks = (qkv_elements + threads_per_block - 1) / threads_per_block;
    reshape_qkv_kernel<<<reshape_q_blocks, threads_per_block>>>(block->q_proj, block->q_reshaped, batch_size, seq, num_q_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());

    size_t kv_elements = (size_t)batch_size * seq * kv_dim;
    int reshape_kv_blocks = (kv_elements + threads_per_block - 1) / threads_per_block;
    reshape_qkv_kernel<<<reshape_kv_blocks, threads_per_block>>>(block->k_proj, block->k_reshaped, batch_size, seq, num_kv_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());
    reshape_qkv_kernel<<<reshape_kv_blocks, threads_per_block>>>(block->v_proj, block->v_reshaped, batch_size, seq, num_kv_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());

    // 3. Calculate Attention Scores (Q @ K^T / sqrt(head_dim))
    // This requires batched GEMM or a custom kernel. Let's use a custom kernel.
    // Output: attn_scores [batch, num_q_heads, seq, seq]
    dim3 score_threads(32, 8); // 32 for target seq (i), 8 for source seq (j) -> 256 threads
    dim3 score_blocks( (seq + score_threads.x - 1) / score_threads.x, num_q_heads, batch_size);
    attention_scores_kernel<<<score_blocks, score_threads>>>(block->q_reshaped, block->k_reshaped, block->attn_scores,
                                                            batch_size, seq, num_q_heads, num_kv_heads, head_dim, q_heads_per_kv);
    CHECK_CUDA(cudaGetLastError());

    // 4. Apply Causal Mask
    dim3 mask_threads(32, 8); // 32 for target seq (i), 8 for source seq (j)
    dim3 mask_blocks( (seq + mask_threads.x - 1) / mask_threads.x, num_q_heads, batch_size);
    apply_causal_mask_kernel<<<mask_blocks, mask_threads>>>(block->attn_scores, batch_size, num_q_heads, seq);
    CHECK_CUDA(cudaGetLastError());

    // 5. Softmax
    // Input/Output: attn_scores -> attn_softmax [batch, num_q_heads, seq, seq]
    // Kernel operates row-wise. Each block handles one row (b, h, i).
    int softmax_threads = 256; // Threads per row
    dim3 softmax_blocks(seq, num_q_heads, batch_size);
    size_t softmax_shmem = seq * sizeof(float); // Shared memory per block
    CHECK_CUDA(cudaMemcpy(block->attn_softmax, block->attn_scores, (size_t)batch_size * num_q_heads * seq * seq * sizeof(float), cudaMemcpyDeviceToDevice)); // Copy scores to softmax buffer
    softmax_forward_kernel<<<softmax_blocks, softmax_threads, softmax_shmem>>>(block->attn_softmax, batch_size, num_q_heads, seq);
    CHECK_CUDA(cudaGetLastError());


    // 6. Calculate Weighted Values (Attention @ V)
    // Input: attn_softmax [b, nq, s, s], V_reshaped [b, nkv, s, hd]
    // Output: attn_output_reshaped [b, nq, s, hd]
    // Custom kernel needed.
    dim3 value_threads(head_dim); // Each thread computes one dimension 'd' for a given (b, hq, i)
    dim3 value_blocks(seq, num_q_heads, batch_size);
    attention_values_kernel<<<value_blocks, value_threads>>>(block->attn_softmax, block->v_reshaped, block->attn_output_reshaped,
                                                           batch_size, seq, num_q_heads, num_kv_heads, head_dim, q_heads_per_kv);
    CHECK_CUDA(cudaGetLastError());

    // 7. Reshape Attention Output
    // [b, nq, s, hd] -> [bs, nq * hd] = [bs, embed]
    int reshape_out_blocks = (total_bse + threads_per_block - 1) / threads_per_block;
    reshape_output_kernel<<<reshape_out_blocks, threads_per_block>>>(block->attn_output_reshaped, block->attn_output, batch_size, seq, num_q_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());

    // 8. Output Projection (GEMM)
    // Input: attn_output [bs, o_dim], Weight: o_proj_weight [o_dim, embed]
    // Output: gqa_output [bs, embed]
    // C = A @ B -> C=gqa_output(bs, embed), A=attn_output(bs, o_dim), B=o_proj_weight(o_dim, embed)
    // M=bs, N=embed, K=o_dim
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             embed, combined_batch_seq, o_dim, // N, M, K
                             &alpha,
                             block->o_proj_weight, embed,     // B (lda=N)
                             block->attn_output, o_dim,       // A (lda=K)
                             &beta,
                             block->gqa_output, embed));        // C (lda=N)

    // 9. Add Residual 1 (Input to GQA)
    // output_after_gqa = gqa_output + input
    // Store result in block->residual (used as input to next RMSNorm)
    CHECK_CUDA(cudaMemcpy(block->residual, block->input_buffer, total_bse * sizeof(float), cudaMemcpyDeviceToDevice)); // Copy original input
    CHECK_CUBLAS(cublasSaxpy(handle, total_bse, &alpha, block->gqa_output, 1, block->residual, 1)); // Add GQA output

    // --- Channel Mixing ---

    // 10. RMSNorm 2
    rmsnorm_forward_kernel<<<rms_blocks, threads_per_block>>>(block->residual, block->normalized2, block->rms_vars2,
                                                block->rmsnorm2_weight, batch_size, seq, embed);
    CHECK_CUDA(cudaGetLastError());

    // 11. Up-projection GEMM: channel_up_output = normalized2 @ channel_up_weight^T
    // C = A @ B -> C=up_out(bs, hidden), A=norm2(bs, embed), B=up_weight(embed, hidden)
    // M=bs, N=hidden, K=embed
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             hidden_mlp, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->channel_up_weight, hidden_mlp, // B (lda=N)
                             block->normalized2, embed,           // A (lda=K)
                             &beta,
                             block->channel_up_output, hidden_mlp)); // C (lda=N)

    // 12. SiLU Activation
    int silu_blocks = (total_bsh + threads_per_block - 1) / threads_per_block;
    silu_kernel<<<silu_blocks, threads_per_block>>>(block->channel_up_output, block->channel_up_activated, total_bsh);
    CHECK_CUDA(cudaGetLastError());

    // 13. Down-projection GEMM: final_output = channel_up_activated @ channel_down_weight^T
    // C = A @ B -> C=output(bs, embed), A=up_act(bs, hidden), B=down_weight(hidden, embed)
    // M=bs, N=embed, K=hidden
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             embed, combined_batch_seq, hidden_mlp, // N, M, K
                             &alpha,
                             block->channel_down_weight, embed,       // B (lda=N)
                             block->channel_up_activated, hidden_mlp, // A (lda=K)
                             &beta,
                             output, embed));                         // C (lda=N) - Write directly to final output

    // 14. Add Residual 2 (Input to Channel Mixing)
    // output = output + residual (residual = output of GQA + residual1)
    CHECK_CUBLAS(cublasSaxpy(handle, total_bse, &alpha, block->residual, 1, output, 1));
}


// Backward pass through one MixerBlock with GQA
void mixer_block_backward(MixerBlock* block, float* d_output, float* d_input, int batch_size, cublasHandle_t handle) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int hidden_mlp = block->hidden_dim;
    int num_q_heads = block->num_q_heads;
    int num_kv_heads = block->num_kv_heads;
    int head_dim = block->head_dim;
    int q_heads_per_kv = block->q_heads_per_kv;

    size_t total_bse = (size_t)batch_size * seq * embed;
    size_t total_bsh = (size_t)batch_size * seq * hidden_mlp;
    size_t total_bs = (size_t)batch_size * seq;

    float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f; // For GEMM accumulation
    int threads_per_block = 256;
    int combined_batch_seq = batch_size * seq;


    // Save incoming gradient (gradient w.r.t. the block's final output)
    CHECK_CUDA(cudaMemcpy(block->d_output_block, d_output, total_bse * sizeof(float), cudaMemcpyDeviceToDevice));

    // Gradient flows into d_input initially, which is the sum of gradients from
    // the channel mixing output and the second residual connection.
    // d_input = d_output (from chain rule for residual y = x + f(x))
    CHECK_CUDA(cudaMemcpy(d_input, d_output, total_bse * sizeof(float), cudaMemcpyDeviceToDevice));
    // d_channel_mixed = d_output (gradient w.r.t output of down-proj GEMM)

    // --- Channel Mixing Backward ---

    // 1. Backprop through Down-projection GEMM
    // Output = channel_up_activated @ channel_down_weight^T
    // d_channel_down_weight = channel_up_activated^T @ d_output
    // d_channel_up_activated = d_output @ channel_down_weight
    // --- Grad Weight ---
    // C = A^T @ B -> C=dW(hidden, embed), A=up_act(bs, hidden), B=d_output(bs, embed)
    // M=hidden, N=embed, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             embed, hidden_mlp, combined_batch_seq, // N, M, K
                             &alpha,
                             d_output, embed,                     // B^T (lda=N)
                             block->channel_up_activated, hidden_mlp, // A^T (lda=K)
                             &beta, // Overwrite gradient
                             block->channel_down_weight_grad, embed)); // C (lda=N)
    // --- Grad Input ---
    // C = A @ B -> C=d_up_act(bs, hidden), A=d_output(bs, embed), B=down_weight(hidden, embed)^T
    // M=bs, N=hidden, K=embed
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             hidden_mlp, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->channel_down_weight, embed,      // B (lda=K)
                             d_output, embed,                      // A (lda=K)
                             &beta,
                             block->d_channel_up_activated, hidden_mlp)); // C (lda=N)

    // 2. Backprop through SiLU Activation
    int silu_blocks = (total_bsh + threads_per_block - 1) / threads_per_block;
    silu_deriv_mult_kernel<<<silu_blocks, threads_per_block>>>(block->channel_up_output, block->d_channel_up_activated,
                                                            block->d_channel_up_output, total_bsh);
    CHECK_CUDA(cudaGetLastError());

    // 3. Backprop through Up-projection GEMM
    // channel_up_output = normalized2 @ channel_up_weight^T
    // d_channel_up_weight = normalized2^T @ d_channel_up_output
    // d_normalized2 = d_channel_up_output @ channel_up_weight
     // --- Grad Weight ---
    // C = A^T @ B -> C=dW(embed, hidden), A=norm2(bs, embed), B=d_up_out(bs, hidden)
    // M=embed, N=hidden, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             hidden_mlp, embed, combined_batch_seq, // N, M, K
                             &alpha,
                             block->d_channel_up_output, hidden_mlp, // B^T (lda=N)
                             block->normalized2, embed,             // A^T (lda=K)
                             &beta, // Overwrite gradient
                             block->channel_up_weight_grad, hidden_mlp)); // C (lda=N)
    // --- Grad Input ---
    // C = A @ B -> C=d_norm2(bs, embed), A=d_up_out(bs, hidden), B=up_weight(embed, hidden)^T
    // M=bs, N=embed, K=hidden
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             embed, combined_batch_seq, hidden_mlp, // N, M, K
                             &alpha,
                             block->channel_up_weight, hidden_mlp,     // B (lda=K)
                             block->d_channel_up_output, hidden_mlp, // A (lda=K)
                             &beta,
                             block->d_normalized2, embed));           // C (lda=N)

    // 4. Backprop through RMSNorm 2
    // Needs: input=residual, grad_out=d_normalized2, vars=rms_vars2, weight=rmsnorm2_weight
    // Outputs: grad_in=d_residual, grad_weight=rmsnorm2_weight_grad
    int rms_blocks = (batch_size * seq + threads_per_block - 1) / threads_per_block;
    size_t rms_shmem = embed * sizeof(float);
    CHECK_CUDA(cudaMemset(block->rmsnorm2_weight_grad, 0, embed * sizeof(float))); // Zero gradient buffer first
    rmsnorm_backward_kernel<<<rms_blocks, threads_per_block, rms_shmem>>>(
        block->residual, block->d_normalized2, block->rms_vars2,
        block->rmsnorm2_weight, block->d_residual, block->rmsnorm2_weight_grad,
        batch_size, seq, embed
    );
    CHECK_CUDA(cudaGetLastError());

    // 5. Add gradient from Residual 2 connection
    // d_residual += d_output_block (gradient flowing from the final output)
    CHECK_CUBLAS(cublasSaxpy(handle, total_bse, &alpha, block->d_output_block, 1, block->d_residual, 1));
    // Now block->d_residual contains the gradient w.r.t the output of (GQA + Residual 1)

    // --- Grouped Query Attention Backward ---
    // Gradient w.r.t GQA output = d_residual
    CHECK_CUDA(cudaMemcpy(block->d_gqa_output, block->d_residual, total_bse * sizeof(float), cudaMemcpyDeviceToDevice));

    // 1. Backprop through Output Projection
    // gqa_output = attn_output @ o_proj_weight^T
    // d_o_proj_weight = attn_output^T @ d_gqa_output
    // d_attn_output = d_gqa_output @ o_proj_weight
    int q_dim = num_q_heads * head_dim; // = o_dim
    int o_dim = q_dim;
     // --- Grad Weight ---
    // C = A^T @ B -> C=dW(o_dim, embed), A=attn_out(bs, o_dim), B=d_gqa_out(bs, embed)
    // M=o_dim, N=embed, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             embed, o_dim, combined_batch_seq, // N, M, K
                             &alpha,
                             block->d_gqa_output, embed,      // B^T (lda=N)
                             block->attn_output, o_dim,       // A^T (lda=K)
                             &beta, // Overwrite
                             block->o_proj_weight_grad, embed)); // C (lda=N)
     // --- Grad Input ---
    // C = A @ B -> C=d_attn_out(bs, o_dim), A=d_gqa_out(bs, embed), B=o_proj_weight(o_dim, embed)^T
    // M=bs, N=o_dim, K=embed
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             o_dim, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             block->o_proj_weight, embed,      // B (lda=K)
                             block->d_gqa_output, embed,      // A (lda=K)
                             &beta,
                             block->d_attn_output, o_dim));     // C (lda=N)

    // 2. Backprop through Reshape Output
    // [bs, nq * hd] -> [b, nq, s, hd]
    int reshape_out_blocks = (total_bse + threads_per_block - 1) / threads_per_block;
    reshape_output_backward_kernel<<<reshape_out_blocks, threads_per_block>>>(block->d_attn_output, block->d_attn_output_reshaped,
                                                                            batch_size, seq, num_q_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());

    // 3. Backprop through Attention Values (Attention @ V)
    // Computes d_attn_softmax and d_v_reshaped
    // Inputs: d_attn_output_reshaped, attn_softmax, v_reshaped
    dim3 value_bwd_threads(32, 8); // Example for d_softmax part
    dim3 value_bwd_blocks( (seq + value_bwd_threads.x - 1) / value_bwd_threads.x, num_q_heads, batch_size);
    // Need separate launch config potentially for dV calculation or handle within kernel
    CHECK_CUDA(cudaMemset(block->d_v_reshaped, 0, (size_t)batch_size * num_kv_heads * seq * head_dim * sizeof(float))); // Zero accumulator
    attention_values_backward_kernel<<<value_bwd_blocks, value_bwd_threads>>>(block->d_attn_output_reshaped, block->attn_softmax, block->v_reshaped,
                                                                             block->d_attn_softmax, block->d_v_reshaped,
                                                                             batch_size, seq, num_q_heads, num_kv_heads, head_dim, q_heads_per_kv);
    CHECK_CUDA(cudaGetLastError());


    // 4. Backprop through Softmax
    // Input: d_attn_softmax, attn_softmax -> Output: d_attn_scores
    int softmax_threads = 256;
    dim3 softmax_blocks(seq, num_q_heads, batch_size);
    size_t softmax_shmem = seq * sizeof(float); // Shared memory per block
    softmax_backward_kernel<<<softmax_blocks, softmax_threads, softmax_shmem>>>(block->d_attn_softmax, block->attn_softmax, block->d_attn_scores,
                                                                               batch_size, num_q_heads, seq);
    CHECK_CUDA(cudaGetLastError());
    // Note: Causal mask gradient is implicitly handled by softmax backward if masked values were -inf -> softmax 0.

    // 5. Backprop through Attention Scores (Q @ K^T / scale)
    // Computes d_q_reshaped and d_k_reshaped
    // Inputs: d_attn_scores, q_reshaped, k_reshaped
    dim3 score_bwd_threads(head_dim); // Thread per head dim 'd'
    dim3 score_bwd_blocks(seq, num_q_heads, batch_size); // Block per (b, hq, i)
     // Need separate launch config potentially for dK calculation or handle within kernel
    CHECK_CUDA(cudaMemset(block->d_k_reshaped, 0, (size_t)batch_size * num_kv_heads * seq * head_dim * sizeof(float))); // Zero accumulator
    attention_scores_backward_kernel<<<score_bwd_blocks, score_bwd_threads>>>(block->d_attn_scores, block->q_reshaped, block->k_reshaped,
                                                                            block->d_q_reshaped, block->d_k_reshaped,
                                                                            batch_size, seq, num_q_heads, num_kv_heads, head_dim, q_heads_per_kv);
    CHECK_CUDA(cudaGetLastError());

    // 6. Backprop through Reshape Q, K, V
    // Inputs: d_q_reshaped, d_k_reshaped, d_v_reshaped
    // Outputs: d_q_proj, d_k_proj, d_v_proj
    int reshape_qkv_bwd_blocks = (total_bse > (size_t)batch_size * seq * num_kv_heads * head_dim * 2) ?
                                 (total_bse + threads_per_block -1) / threads_per_block :
                                 ((size_t)batch_size * seq * num_kv_heads * head_dim * 2 + threads_per_block - 1) / threads_per_block;

    reshape_qkv_backward_kernel<<<reshape_qkv_bwd_blocks, threads_per_block>>>(block->d_q_reshaped, block->d_k_reshaped, block->d_v_reshaped,
                                                                              block->d_q_proj, block->d_k_proj, block->d_v_proj,
                                                                              batch_size, seq, num_q_heads, num_kv_heads, head_dim);
    CHECK_CUDA(cudaGetLastError());

    // 7. Backprop through Q, K, V Projections
    // Q = normalized1 @ q_proj_weight^T
    // d_q_proj_weight = normalized1^T @ d_q_proj
    // d_normalized1_from_q = d_q_proj @ q_proj_weight
    int kv_dim = num_kv_heads * head_dim;
    // --- Grad Weights ---
    // dWq = A^T @ B -> C=dWq(embed, q_dim), A=norm1(bs, embed), B=d_q_proj(bs, q_dim) M=embed, N=q_dim, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             q_dim, embed, combined_batch_seq, // N, M, K
                             &alpha,
                             block->d_q_proj, q_dim,           // B^T (lda=N)
                             block->normalized1, embed,       // A^T (lda=K)
                             &beta, // Overwrite
                             block->q_proj_weight_grad, q_dim)); // C (lda=N)

    // dWk = norm1^T @ d_k_proj -> M=embed, N=kv_dim, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             kv_dim, embed, combined_batch_seq, // N, M, K
                             &alpha,
                             block->d_k_proj, kv_dim,          // B^T
                             block->normalized1, embed,       // A^T
                             &beta, // Overwrite
                             block->k_proj_weight_grad, kv_dim)); // C

    // dWv = norm1^T @ d_v_proj -> M=embed, N=kv_dim, K=bs
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             kv_dim, embed, combined_batch_seq, // N, M, K
                             &alpha,
                             block->d_v_proj, kv_dim,          // B^T
                             block->normalized1, embed,       // A^T
                             &beta, // Overwrite
                             block->v_proj_weight_grad, kv_dim)); // C

    // --- Grad Input (d_normalized1) ---
    // Accumulate gradients from Q, K, V paths
    // d_norm1 = d_q_proj @ q_proj_weight -> C(bs, embed), A(bs, q_dim), B(q_dim, embed)^T M=bs, N=embed, K=q_dim
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             embed, combined_batch_seq, q_dim, // N, M, K
                             &alpha,
                             block->q_proj_weight, q_dim,      // B (lda=K)
                             block->d_q_proj, q_dim,           // A (lda=K)
                             &beta, // Start with dQ contribution
                             block->d_normalized1, embed));    // C (lda=N)

    // d_norm1 += d_k_proj @ k_proj_weight -> M=bs, N=embed, K=kv_dim
     CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             embed, combined_batch_seq, kv_dim, // N, M, K
                             &alpha, // Use alpha=1 to add
                             block->k_proj_weight, kv_dim,     // B
                             block->d_k_proj, kv_dim,          // A
                             &beta_one, // Accumulate
                             block->d_normalized1, embed));   // C

     // d_norm1 += d_v_proj @ v_proj_weight -> M=bs, N=embed, K=kv_dim
     CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             embed, combined_batch_seq, kv_dim, // N, M, K
                             &alpha, // Use alpha=1 to add
                             block->v_proj_weight, kv_dim,     // B
                             block->d_v_proj, kv_dim,          // A
                             &beta_one, // Accumulate
                             block->d_normalized1, embed));   // C

    // 8. Backprop through RMSNorm 1
    // Needs: input=input_buffer, grad_out=d_normalized1, vars=rms_vars1, weight=rmsnorm1_weight
    // Outputs: grad_in=d_input (final grad w.r.t block input before residual), grad_weight=rmsnorm1_weight_grad
    CHECK_CUDA(cudaMemset(block->rmsnorm1_weight_grad, 0, embed * sizeof(float))); // Zero gradient buffer first
    rmsnorm_backward_kernel<<<rms_blocks, threads_per_block, rms_shmem>>>(
        block->input_buffer, block->d_normalized1, block->rms_vars1,
        block->rmsnorm1_weight, d_input, block->rmsnorm1_weight_grad, // Write grad_in directly to final d_input
        batch_size, seq, embed
    );
     CHECK_CUDA(cudaGetLastError());

    // 9. Add gradient from Residual 1 connection
    // d_input += d_residual (gradient flowing back from the channel mixing part)
    CHECK_CUBLAS(cublasSaxpy(handle, total_bse, &alpha, block->d_residual, 1, d_input, 1));
    // d_input now holds the final gradient w.r.t. the block's input
}


// Forward pass through the entire model.
void mixer_model_forward(MixerModel* model, int* d_input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    size_t tensor_elements = (size_t)batch * seq * embed;
    float alpha = 1.0f, beta = 0.0f;

    // 1. Embedding lookup
    int threads = 256;
    int embed_blocks = (batch * seq + threads - 1) / threads;
    apply_embedding_kernel<<<embed_blocks, threads>>>(d_input_tokens, model->embedding_weight, model->embeddings,
                                                 batch, seq, embed, model->vocab_size);
     CHECK_CUDA(cudaGetLastError());

    // 2. Copy embeddings to the first slot of block_outputs (input to layer 0)
    float* current_input = model->block_outputs; // Points to start of block_outputs[0]
    CHECK_CUDA(cudaMemcpy(current_input, model->embeddings, tensor_elements * sizeof(float), cudaMemcpyDeviceToDevice));

    // 3. Forward through each MixerBlock
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        mixer_block_forward(model->blocks[i], input_ptr, output_ptr, batch, model->cublas_handle);
    }

    // 4. Output projection: logits = final_output @ out_proj_weight^T
    float* final_output = model->block_outputs + model->num_layers * tensor_elements; // Output of the last layer
    int combined_batch_seq = batch * seq;
    // C = A @ B -> C=logits(bs, vocab), A=final_out(bs, embed), B=out_proj_weight(embed, vocab)
    // M=bs, N=vocab, K=embed
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             model->vocab_size, combined_batch_seq, embed, // N, M, K
                             &alpha,
                             model->out_proj_weight, model->vocab_size, // B (lda=N)
                             final_output, embed,                      // A (lda=K)
                             &beta,
                             model->logits, model->vocab_size));         // C (lda=N)
}

// Compute cross-entropy loss and initial gradients w.r.t logits
float compute_loss_and_gradients(MixerModel* model, int* d_target_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int vocab = model->vocab_size;
    size_t total_tokens = (size_t)batch * seq;

    // Launch kernel to compute softmax, loss, and d_logits in one go
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;
    softmax_cross_entropy_kernel<<<blocks, threads>>>(
        model->logits, d_target_tokens,
        model->d_logits, model->d_loss_buffer,
        batch, seq, vocab
    );
     CHECK_CUDA(cudaGetLastError());

    // Retrieve loss values to host
    CHECK_CUDA(cudaMemcpy(model->h_loss_buffer, model->d_loss_buffer, total_tokens * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum up the loss on CPU
    double total_loss = 0.0; // Use double for accumulation
    for (size_t i = 0; i < total_tokens; i++) {
        total_loss += model->h_loss_buffer[i];
    }

    // Return average loss per token
    return (float)(total_loss / total_tokens);
}

// Backward pass through the entire model.
void mixer_model_backward(MixerModel* model) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t tensor_elements = (size_t)batch * seq * embed;
    int combined_batch_seq = batch * seq;
    float alpha = 1.0f, beta = 0.0f;

    // 1. Backprop through Output Projection
    // logits = final_output @ out_proj_weight^T
    // d_out_proj_weight = final_output^T @ d_logits
    // d_final_output = d_logits @ out_proj_weight
    float* final_output = model->block_outputs + model->num_layers * tensor_elements;
    float* d_final_output = model->d_block_outputs + model->num_layers * tensor_elements; // Grad w.r.t last layer output

    // --- Grad Weight ---
    // C = A^T @ B -> C=dW(embed, vocab), A=final_out(bs, embed), B=d_logits(bs, vocab)
    // M=embed, N=vocab, K=bs
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             vocab, embed, combined_batch_seq, // N, M, K
                             &alpha,
                             model->d_logits, vocab,          // B^T (lda=N)
                             final_output, embed,             // A^T (lda=K)
                             &beta, // Overwrite
                             model->out_proj_weight_grad, vocab)); // C (lda=N)

    // --- Grad Input ---
    // C = A @ B -> C=d_final(bs, embed), A=d_logits(bs, vocab), B=out_proj_weight(embed, vocab)^T
    // M=bs, N=embed, K=vocab
    CHECK_CUBLAS(cublasSgemm(model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             embed, combined_batch_seq, vocab, // N, M, K
                             &alpha,
                             model->out_proj_weight, vocab,    // B (lda=K)
                             model->d_logits, vocab,          // A (lda=K)
                             &beta,
                             d_final_output, embed));         // C (lda=N)


    // 2. Backprop through Mixer blocks
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output_layer = model->d_block_outputs + (i + 1) * tensor_elements; // Grad from layer above
        float* d_input_layer = model->d_block_outputs + i * tensor_elements;     // Grad w.r.t input of this layer
        mixer_block_backward(model->blocks[i], d_output_layer, d_input_layer, batch, model->cublas_handle);
    }

    // 3. Backprop through Embedding
    // d_input_layer now points to d_block_outputs[0], which is the gradient w.r.t the embeddings
    float* d_embeddings = model->d_block_outputs; // Alias for clarity
    size_t embed_matrix_size = (size_t)model->vocab_size * embed * sizeof(float);

    // Zero out the embedding gradient first (important due to atomic adds)
    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, embed_matrix_size));

    // Launch kernel to accumulate gradients
    int threads = 256;
    int blocks = (batch * seq + threads - 1) / threads;
    embedding_backward_kernel<<<blocks, threads>>>(
        model->d_input_tokens, d_embeddings,
        model->embedding_weight_grad, batch, seq, embed, model->vocab_size
    );
     CHECK_CUDA(cudaGetLastError());
}

// Update weights using AdamW optimizer
void update_weights_adamw(MixerModel* model, float learning_rate) {
    model->t++;  // Increment time step.

    // Calculate bias correction factors and effective learning rate for this step
    float beta1_pow_t = powf(model->beta1, model->t);
    float beta2_pow_t = powf(model->beta2, model->t);
    // Effective LR = LR * sqrt(1 - beta2^t) / (1 - beta1^t)
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_pow_t) / (1.0f - beta1_pow_t);

    // Scale factor for gradient averaging (average gradient over batch * seq_length)
    float scale = (float)model->batch_size * model->seq_length;
    if (scale == 0) scale = 1.0f; // Avoid division by zero

    int threads = 256;
    int blocks;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t vocab_embed_elements = (size_t)vocab * embed;

    // --- Update Embedding Weights ---
    blocks = (vocab_embed_elements + threads - 1) / threads;
    adamw_update_kernel<<<blocks, threads>>>(model->embedding_weight, model->embedding_weight_grad,
                                            model->embedding_m, model->embedding_v,
                                            model->beta1, model->beta2, model->epsilon,
                                            learning_rate, model->weight_decay, alpha_t,
                                            vocab_embed_elements, scale);
     CHECK_CUDA(cudaGetLastError());

    // --- Update Output Projection Weights ---
    blocks = (vocab_embed_elements + threads - 1) / threads;
    adamw_update_kernel<<<blocks, threads>>>(model->out_proj_weight, model->out_proj_weight_grad,
                                            model->out_proj_m, model->out_proj_v,
                                            model->beta1, model->beta2, model->epsilon,
                                            learning_rate, model->weight_decay, alpha_t,
                                            vocab_embed_elements, scale);
    CHECK_CUDA(cudaGetLastError());

    // --- Update MixerBlock Weights ---
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int q_dim = block->num_q_heads * block->head_dim;
        int kv_dim = block->num_kv_heads * block->head_dim;
        int o_dim = q_dim;
        int hidden_mlp = block->hidden_dim;

        size_t size_q_proj = (size_t)embed * q_dim;
        size_t size_kv_proj = (size_t)embed * kv_dim;
        size_t size_o_proj = (size_t)o_dim * embed;
        size_t size_up = (size_t)embed * hidden_mlp;
        size_t size_down = (size_t)hidden_mlp * embed;
        size_t size_norm = embed;

        // GQA Weights
        blocks = (size_q_proj + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->q_proj_weight, block->q_proj_weight_grad, block->q_proj_m, block->q_proj_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_q_proj, scale);
        CHECK_CUDA(cudaGetLastError());
        blocks = (size_kv_proj + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->k_proj_weight, block->k_proj_weight_grad, block->k_proj_m, block->k_proj_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_kv_proj, scale);
        CHECK_CUDA(cudaGetLastError());
        adamw_update_kernel<<<blocks, threads>>>(block->v_proj_weight, block->v_proj_weight_grad, block->v_proj_m, block->v_proj_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_kv_proj, scale);
         CHECK_CUDA(cudaGetLastError());
        blocks = (size_o_proj + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->o_proj_weight, block->o_proj_weight_grad, block->o_proj_m, block->o_proj_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_o_proj, scale);
        CHECK_CUDA(cudaGetLastError());

        // Channel Mixing Weights
        blocks = (size_up + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->channel_up_weight, block->channel_up_weight_grad, block->channel_up_m, block->channel_up_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_up, scale);
        CHECK_CUDA(cudaGetLastError());
        blocks = (size_down + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->channel_down_weight, block->channel_down_weight_grad, block->channel_down_m, block->channel_down_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_down, scale);
        CHECK_CUDA(cudaGetLastError());

        // RMSNorm Weights
        blocks = (size_norm + threads - 1) / threads;
        adamw_update_kernel<<<blocks, threads>>>(block->rmsnorm1_weight, block->rmsnorm1_weight_grad, block->rmsnorm1_m, block->rmsnorm1_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_norm, scale);
         CHECK_CUDA(cudaGetLastError());
        adamw_update_kernel<<<blocks, threads>>>(block->rmsnorm2_weight, block->rmsnorm2_weight_grad, block->rmsnorm2_m, block->rmsnorm2_v, model->beta1, model->beta2, model->epsilon, learning_rate, model->weight_decay, alpha_t, size_norm, scale);
        CHECK_CUDA(cudaGetLastError());
    }
}

// Zero out all gradients
void zero_gradients(MixerModel* model) {
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t size_vocab_embed = (size_t)vocab * embed * sizeof(float);

    CHECK_CUDA(cudaMemset(model->embedding_weight_grad, 0, size_vocab_embed));
    CHECK_CUDA(cudaMemset(model->out_proj_weight_grad, 0, size_vocab_embed));

    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int q_dim = block->num_q_heads * block->head_dim;
        int kv_dim = block->num_kv_heads * block->head_dim;
        int o_dim = q_dim;
        int hidden_mlp = block->hidden_dim;

        size_t size_q_proj = (size_t)embed * q_dim * sizeof(float);
        size_t size_kv_proj = (size_t)embed * kv_dim * sizeof(float);
        size_t size_o_proj = (size_t)o_dim * embed * sizeof(float);
        size_t size_up = (size_t)embed * hidden_mlp * sizeof(float);
        size_t size_down = (size_t)hidden_mlp * embed * sizeof(float);
        size_t size_norm = (size_t)embed * sizeof(float);

        CHECK_CUDA(cudaMemset(block->q_proj_weight_grad, 0, size_q_proj));
        CHECK_CUDA(cudaMemset(block->k_proj_weight_grad, 0, size_kv_proj));
        CHECK_CUDA(cudaMemset(block->v_proj_weight_grad, 0, size_kv_proj));
        CHECK_CUDA(cudaMemset(block->o_proj_weight_grad, 0, size_o_proj));
        CHECK_CUDA(cudaMemset(block->channel_up_weight_grad, 0, size_up));
        CHECK_CUDA(cudaMemset(block->channel_down_weight_grad, 0, size_down));
        CHECK_CUDA(cudaMemset(block->rmsnorm1_weight_grad, 0, size_norm));
        CHECK_CUDA(cudaMemset(block->rmsnorm2_weight_grad, 0, size_norm));
    }
}

// --- Utility Functions ---

// Load text data from a file
char* load_text_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size_long = ftell(file);
     if (file_size_long < 0) {
        fprintf(stderr, "Error: could not get file size for %s\n", filename);
        fclose(file);
        return NULL;
    }
    fseek(file, 0, SEEK_SET);
     size_t file_size = (size_t)file_size_long;
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: memory allocation failed for text buffer\n");
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if(bytes_read != file_size) {
        fprintf(stderr, "Error: could not read entire file %s (read %zu/%zu bytes)\n", filename, bytes_read, file_size);
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[file_size] = '\0'; // Null-terminate
    *size = file_size;
    fclose(file);
    return buffer;
}

// Get a batch of randomly sampled sequences from the text
void get_random_batch(const char* text, size_t text_size, int batch_size, int seq_length,
                      int* input_tokens, int* target_tokens) {
    // Ensure text is long enough to sample a sequence + target
    if (text_size <= (size_t)seq_length) {
        fprintf(stderr, "Error: Text size (%zu) is not greater than sequence length (%d)\n", text_size, seq_length);
        // Consider padding or other error handling if this is recoverable
         exit(EXIT_FAILURE);
    }

    size_t max_start_pos = text_size - seq_length - 1;

    for (int b = 0; b < batch_size; b++) {
        // Generate random start position
        size_t start_pos = (size_t)rand() % (max_start_pos + 1); // +1 because % can return max_start_pos

        // Fill input and target tokens for this sequence
        int* input_ptr = input_tokens + b * seq_length;
        int* target_ptr = target_tokens + b * seq_length;
        for (int s = 0; s < seq_length; s++) {
            // Cast char to unsigned char before int to handle potential negative char values correctly
            input_ptr[s] = (int)(unsigned char)text[start_pos + s];
            target_ptr[s] = (int)(unsigned char)text[start_pos + s + 1];
        }
    }
}

// Count model parameters.
long long count_parameters(MixerModel* model) { // Use long long for potentially large models
    long long total_params = 0;

    // Embedding and Output Projection
    total_params += (long long)model->vocab_size * model->embed_dim * 2;

    // Parameters per layer
    if (model->num_layers > 0) {
        MixerBlock* block = model->blocks[0]; // Use first block to get dimensions
        int q_dim = block->num_q_heads * block->head_dim;
        int kv_dim = block->num_kv_heads * block->head_dim;
        int o_dim = q_dim;

        long long params_per_layer = 0;
        // GQA parameters
        params_per_layer += (long long)model->embed_dim * q_dim;  // Q proj
        params_per_layer += (long long)model->embed_dim * kv_dim; // K proj
        params_per_layer += (long long)model->embed_dim * kv_dim; // V proj
        params_per_layer += (long long)o_dim * model->embed_dim;  // O proj
        // Channel Mixing parameters
        params_per_layer += (long long)model->embed_dim * model->hidden_dim; // Channel up
        params_per_layer += (long long)model->hidden_dim * model->embed_dim; // Channel down
        // RMSNorm parameters
        params_per_layer += (long long)model->embed_dim * 2; // Norm1 weight, Norm2 weight

        total_params += params_per_layer * model->num_layers;
    }

    return total_params;
}

// Save model weights to a binary file.
void save_model(MixerModel* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        return;
    }

    // --- Write Header ---
    // Versioning (optional but recommended)
    int version = 1; // Simple version number
    fwrite(&version, sizeof(int), 1, file);

    // Dimensions
    fwrite(&model->vocab_size, sizeof(int), 1, file);
    fwrite(&model->embed_dim, sizeof(int), 1, file);
    fwrite(&model->hidden_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    // Save GQA dimensions (added in v1)
    fwrite(&model->num_q_heads, sizeof(int), 1, file);
    fwrite(&model->num_kv_heads, sizeof(int), 1, file);
    // Note: Batch size is runtime, typically not saved. Head dim derived.

    // --- Write Weights ---
    int vocab = model->vocab_size;
    int embed = model->embed_dim;
    size_t vocab_embed_elements = (size_t)vocab * embed;

    // Helper to copy from device and write
    auto write_weights = [&](const float* d_ptr, size_t num_elements) {
        float* h_ptr = (float*)malloc(num_elements * sizeof(float));
        if (!h_ptr) {fprintf(stderr, "Malloc failed during save!\n"); exit(EXIT_FAILURE);}
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
        size_t written = fwrite(h_ptr, sizeof(float), num_elements, file);
        if (written != num_elements) { fprintf(stderr, "fwrite error during save!\n"); exit(EXIT_FAILURE); }
        free(h_ptr);
    };

    // Embedding weights
    write_weights(model->embedding_weight, vocab_embed_elements);

    // Output projection weights
    write_weights(model->out_proj_weight, vocab_embed_elements);

    // MixerBlock weights (layer by layer)
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int q_dim = block->num_q_heads * block->head_dim;
        int kv_dim = block->num_kv_heads * block->head_dim;
        int o_dim = q_dim;
        int hidden_mlp = block->hidden_dim;

        // GQA weights
        write_weights(block->q_proj_weight, (size_t)embed * q_dim);
        write_weights(block->k_proj_weight, (size_t)embed * kv_dim);
        write_weights(block->v_proj_weight, (size_t)embed * kv_dim);
        write_weights(block->o_proj_weight, (size_t)o_dim * embed);

        // Channel mixing weights
        write_weights(block->channel_up_weight, (size_t)embed * hidden_mlp);
        write_weights(block->channel_down_weight, (size_t)hidden_mlp * embed);

        // RMSNorm weights
        write_weights(block->rmsnorm1_weight, embed);
        write_weights(block->rmsnorm2_weight, embed);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from a binary file.
MixerModel* load_model(const char* filename, int batch_size_override) { // Allow overriding batch size at load time
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for reading\n", filename);
        return NULL;
    }

    // --- Read Header ---
    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 1) { // Check compatibility
        fprintf(stderr, "Error: Unsupported model version %d in %s\n", version, filename);
        fclose(file);
        return NULL;
    }

    int vocab_size, embed_dim, hidden_dim, num_layers, seq_length;
    int num_q_heads, num_kv_heads; // Added in v1
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&num_q_heads, sizeof(int), 1, file);
    fread(&num_kv_heads, sizeof(int), 1, file);

    // Initialize a new model with read dimensions and override batch size
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size_override, num_q_heads, num_kv_heads);

     // Helper to read and copy to device
    auto read_weights = [&](float* d_ptr, size_t num_elements) {
        float* h_ptr = (float*)malloc(num_elements * sizeof(float));
         if (!h_ptr) {fprintf(stderr, "Malloc failed during load!\n"); exit(EXIT_FAILURE);}
        size_t read_count = fread(h_ptr, sizeof(float), num_elements, file);
        if (read_count != num_elements) { fprintf(stderr, "fread error during load (expected %zu, got %zu)!\n", num_elements, read_count); exit(EXIT_FAILURE); }
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
        free(h_ptr);
    };

    // --- Read Weights ---
    size_t vocab_embed_elements = (size_t)vocab_size * embed_dim;

    // Embedding weights
    read_weights(model->embedding_weight, vocab_embed_elements);

    // Output projection weights
    read_weights(model->out_proj_weight, vocab_embed_elements);

    // MixerBlock weights (layer by layer)
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        // Ensure block dimensions match loaded model dimensions
        if (block->embed_dim != embed_dim || block->seq_length != seq_length || block->num_q_heads != num_q_heads || block->num_kv_heads != num_kv_heads) {
             fprintf(stderr, "Mismatch in loaded block dimensions!\n"); exit(EXIT_FAILURE);
         }
        int q_dim = block->num_q_heads * block->head_dim;
        int kv_dim = block->num_kv_heads * block->head_dim;
        int o_dim = q_dim;
        int hidden_mlp = block->hidden_dim; // Use block's hidden dim

        // GQA weights
        read_weights(block->q_proj_weight, (size_t)embed_dim * q_dim);
        read_weights(block->k_proj_weight, (size_t)embed_dim * kv_dim);
        read_weights(block->v_proj_weight, (size_t)embed_dim * kv_dim);
        read_weights(block->o_proj_weight, (size_t)o_dim * embed_dim);

        // Channel mixing weights
        read_weights(block->channel_up_weight, (size_t)embed_dim * hidden_mlp);
        read_weights(block->channel_down_weight, (size_t)hidden_mlp * embed_dim);

        // RMSNorm weights
        read_weights(block->rmsnorm1_weight, embed_dim);
        read_weights(block->rmsnorm2_weight, embed_dim);
    }

    // Check if end of file is reached (optional sanity check)
    // fgetc(file);
    // if (!feof(file)) {
    //     fprintf(stderr, "Warning: Did not reach end of file after loading weights from %s\n", filename);
    // }

    fclose(file);
    printf("Model loaded from %s\n", filename);
    // Reset optimizer step count when loading
    model->t = 0;
    return model;
}


// Generate text from the model with temperature-based sampling
void generate_text(MixerModel* model, const char* corpus, size_t corpus_size, int max_new_tokens, float temperature) {
    int seq_length = model->seq_length;
    int batch_size = 1; // Generation uses batch size 1

    // Check if model batch size is compatible (should be >= 1)
    if (model->batch_size < 1) {
        fprintf(stderr, "Error: Model batch size (%d) must be >= 1 for generation.\n", model->batch_size);
        return;
    }

    // Allocate host buffer for the single sequence
    int* h_tokens = (int*)malloc(seq_length * sizeof(int));
    if (!h_tokens) { fprintf(stderr, "Failed to allocate token buffer for generation\n"); return; }

    // Get a starting seed sequence from the corpus
    if (corpus_size <= (size_t)seq_length) {
         fprintf(stderr, "Corpus too small for seed sequence.\n"); free(h_tokens); return;
    }
    size_t start_pos = (size_t)rand() % (corpus_size - seq_length);

    printf("--- Seed Context ---\n");
    for (int i = 0; i < seq_length; i++) {
        h_tokens[i] = (int)(unsigned char)corpus[start_pos + i];
        printf("%c", h_tokens[i]);
    }
    printf("\n--- Generated Text ---\n");

    // Use the first slice of the model's input buffer for batch size 1
    int* d_gen_tokens = model->d_input_tokens; // Use the model's existing buffer
    float* d_gen_logits = model->logits;      // Use the model's existing buffer

    // Copy initial tokens to device
    CHECK_CUDA(cudaMemcpy(d_gen_tokens, h_tokens, seq_length * sizeof(int), cudaMemcpyHostToDevice));

    // Generate tokens one by one
    for (int k = 0; k < max_new_tokens; k++) {
        // Forward pass (using only the first batch slot)
        // Note: The model forward pass internally uses model->batch_size, but we only care about the first batch element's logits.
        // This is inefficient if model->batch_size > 1, but avoids reallocating everything.
        // For optimal generation performance, create a separate model instance with batch_size=1.
        mixer_model_forward(model, d_gen_tokens);

        // Get logits for the *last* token position of the *first* batch item
        float* h_logits = (float*)malloc(model->vocab_size * sizeof(float));
        if(!h_logits) {fprintf(stderr, "Malloc failed for logits\n"); break;}

        // Offset to the logits for batch 0, last sequence position (seq_length - 1)
        size_t logits_offset = (size_t)(seq_length - 1) * model->vocab_size;
        CHECK_CUDA(cudaMemcpy(h_logits,
                              d_gen_logits + logits_offset,
                              model->vocab_size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Apply temperature scaling
        float temp = fmaxf(temperature, 1e-9f); // Ensure temperature is positive
        float max_logit = -FLT_MAX;
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] /= temp;
            max_logit = fmaxf(max_logit, h_logits[v]);
        }

        // Compute probabilities (softmax)
        float sum_exp = 0.0f;
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit); // Subtract max for stability
            sum_exp += h_logits[v];
        }
        float inv_sum_exp = 1.0f / (sum_exp + 1e-9f);
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] *= inv_sum_exp; // Normalize to get probabilities
        }

        // Sample from the probability distribution
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
        // If r is exactly 1.0 (rare), next_token might remain 0. Fallback:
        if (r >= cdf) {
           next_token = model->vocab_size - 1; // Pick last token if something went wrong
        }


        // Print the generated character
        printf("%c", (char)next_token);
        fflush(stdout); // Ensure immediate output

        // Update the sequence: shift left and add the new token
        for (int j = 0; j < seq_length - 1; j++) {
            h_tokens[j] = h_tokens[j + 1];
        }
        h_tokens[seq_length - 1] = next_token;

        // Copy updated tokens back to device for the next step
        CHECK_CUDA(cudaMemcpy(d_gen_tokens, h_tokens, seq_length * sizeof(int), cudaMemcpyHostToDevice));

        free(h_logits);
    }

    printf("\n--------------------\n");
    free(h_tokens);
}


// --- Main Function ---
int main(int argc, char** argv) {
    srand((unsigned int)time(NULL)); // Seed random number generator
    CHECK_CUDA(cudaSetDevice(0)); // Select default GPU

    // --- Model Hyperparameters ---
    int vocab_size = 256;      // ASCII
    int embed_dim = 256;       // Embedding dimension
    int num_layers = 12;        // Number of mixer blocks
    int seq_length = 512;     // Sequence length during training/generation
    int batch_size = 32;       // Batch size for training
    int num_q_heads = 8;       // Number of query heads for GQA
    int num_kv_heads = 2;      // Number of key/value heads for GQA (must divide num_q_heads)

    // --- Training Hyperparameters ---
    float learning_rate = 3e-4f;    // Learning rate
    int total_training_steps = 50000; // Total training iterations
    int print_interval = 10;       // Print loss every N steps
    int sample_interval = 1000;     // Generate text sample every N steps
    int save_interval = 10000;      // Save model checkpoint every N steps
    const char* corpus_filename = "combined_corpus.txt"; // Default corpus file
    const char* output_model_basename = "gqa_mixer_model"; // Basename for saved models

    // --- Command Line Args ---
    const char* load_model_path = NULL;
    if (argc > 1) {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            printf("Usage: %s [path_to_load_model.bin]\n", argv[0]);
            printf("Options:\n");
            printf("  [path_to_load_model.bin]: Optional path to a pre-trained model file to resume training or generate text.\n");
             printf("Hyperparameters (set in code):\n");
            printf("  vocab_size=%d, embed_dim=%d, num_layers=%d, seq_length=%d, batch_size=%d\n", vocab_size, embed_dim, num_layers, seq_length, batch_size);
            printf("  num_q_heads=%d, num_kv_heads=%d\n", num_q_heads, num_kv_heads);
            printf("  learning_rate=%.1e, total_steps=%d\n", learning_rate, total_training_steps);
            return 0;
        }
        load_model_path = argv[1];
    }

    // --- Model Initialization ---
    MixerModel* model;
    if (load_model_path) {
        printf("Loading model from: %s\n", load_model_path);
        model = load_model(load_model_path, batch_size); // Load with training batch size
        if (!model) {
            fprintf(stderr, "Failed to load model. Exiting.\n");
            return EXIT_FAILURE;
        }
        // Update dimensions from loaded model if necessary (though init should match)
        vocab_size = model->vocab_size;
        embed_dim = model->embed_dim;
        num_layers = model->num_layers;
        seq_length = model->seq_length;
        num_q_heads = model->num_q_heads;
        num_kv_heads = model->num_kv_heads;
         printf("Model loaded. Resuming training or generating...\n");
    } else {
        printf("Initializing new model...\n");
        model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size, num_q_heads, num_kv_heads);
         printf("New model initialized.\n");
    }

    printf("Model Parameters: %lld\n", count_parameters(model));

    // --- Load Data ---
    size_t text_size;
    char* text = load_text_file(corpus_filename, &text_size);
    if (!text) {
        fprintf(stderr, "Failed to load corpus file: %s. Create a text file named '%s' in the execution directory.\n", corpus_filename, corpus_filename);
         free_mixer_model(model);
        return EXIT_FAILURE;
    }
    printf("Loaded text corpus '%s' with %zu bytes (%.2f MB)\n", corpus_filename, text_size, (double)text_size / (1024*1024));
    if (text_size <= (size_t)seq_length) {
        fprintf(stderr, "Error: Corpus size is too small for sequence length %d.\n", seq_length);
        free(text);
        free_mixer_model(model);
        return EXIT_FAILURE;
    }


    // --- Training Loop ---
    printf("\n--- Starting Training ---\n");
    printf("Hyperparameters: LR=%.1e, Steps=%d, Batch=%d, SeqLen=%d\n",
           learning_rate, total_training_steps, batch_size, seq_length);
    printf("Layers=%d, Embed=%d, HeadsQ=%d, HeadsKV=%d\n",
           num_layers, embed_dim, num_q_heads, num_kv_heads);

    // Allocate host buffers for batches
    int* h_input_tokens = (int*)malloc((size_t)batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc((size_t)batch_size * seq_length * sizeof(int));
     if (!h_input_tokens || !h_target_tokens) {
        fprintf(stderr, "Failed to allocate host token buffers.\n");
        free(text); free_mixer_model(model); return EXIT_FAILURE;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float total_time_ms = 0;
    int steps_since_last_print = 0;

    time_t training_start_time = time(NULL);

    for (int step = model->t; step < total_training_steps; step++) { // Start from model->t if loaded
        CHECK_CUDA(cudaEventRecord(start));

        // 1. Get Batch
        get_random_batch(text, text_size, batch_size, seq_length, h_input_tokens, h_target_tokens);

        // 2. Copy to Device
        size_t batch_token_bytes = (size_t)batch_size * seq_length * sizeof(int);
        CHECK_CUDA(cudaMemcpy(model->d_input_tokens, h_input_tokens, batch_token_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(model->d_target_tokens, h_target_tokens, batch_token_bytes, cudaMemcpyHostToDevice));

        // 3. Zero Gradients (Do this BEFORE backward pass)
        zero_gradients(model); // Moved here for clarity, could be after update

        // 4. Forward Pass
        mixer_model_forward(model, model->d_input_tokens);

        // 5. Compute Loss and Initial Gradients (d_logits)
        float step_loss = compute_loss_and_gradients(model, model->d_target_tokens);

        // 6. Backward Pass
        mixer_model_backward(model);

        // 7. Update Weights (AdamW)
        update_weights_adamw(model, learning_rate);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float step_time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&step_time_ms, start, stop));
        total_time_ms += step_time_ms;
        steps_since_last_print++;

        // Print progress
        if ((step + 1) % print_interval == 0) {
            float avg_step_time = total_time_ms / steps_since_last_print;
            float tokens_per_sec = (float)(batch_size * seq_length * steps_since_last_print) / (total_time_ms / 1000.0f);
            time_t current_time = time(NULL);
            double elapsed_sec = difftime(current_time, training_start_time);

            printf("Step %d/%d | Loss: %.4f | Step Time: %.2f ms | Tokens/sec: %.0f | Elapsed: %.0fs\n",
                   step + 1, total_training_steps, step_loss, avg_step_time, tokens_per_sec, elapsed_sec);

            total_time_ms = 0; // Reset timer for next interval
            steps_since_last_print = 0;
        }

        // Generate sample text
        if ((step + 1) % sample_interval == 0 && step > 0) {
            printf("\n--- Generating Sample @ Step %d ---\n", step + 1);
            // Need to run generation with batch_size=1, potentially requires modifying model or using a separate instance.
            // For simplicity here, we'll just use the existing model, acknowledging inefficiency.
            generate_text(model, text, text_size, 128, 0.8f); // Generate 128 tokens with temp 0.8
        }

         // Save model checkpoint
        if ((step + 1) % save_interval == 0 && step > 0) {
            char model_fname[128];
            snprintf(model_fname, sizeof(model_fname), "%s_step%d.bin", output_model_basename, step + 1);
            save_model(model, model_fname);
        }
    }

    printf("\n--- Training Finished ---\n");

    // --- Save Final Model ---
    char final_model_fname[128];
     time_t now = time(NULL);
     struct tm *tminfo = localtime(&now);
     strftime(final_model_fname, sizeof(final_model_fname), "%Y%m%d_%H%M%S_gqa_mixer_final.bin", tminfo);
    save_model(model, final_model_fname);

    // --- Final Sample Generation ---
     printf("\n--- Final Sample Generation ---\n");
     generate_text(model, text, text_size, 512, 0.7f);


    // --- Cleanup ---
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input_tokens);
    free(h_target_tokens);
    free(text);
    free_mixer_model(model);

    printf("\nDone.\n");
    return 0;
}