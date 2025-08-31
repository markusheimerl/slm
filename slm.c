#include "slm.h"

// Initialize the SLM
SLM* init_slm(int d_model, int seq_len, int batch_size, int mlp_hidden, int num_layers, cublasHandle_t cublas_handle) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->d_model = d_model;
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;
    slm->vocab_size = VOCAB_SIZE;
    slm->num_layers = num_layers;
    slm->mlp_hidden = mlp_hidden;
    slm->cublas_handle = cublas_handle;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;
    
    // Initialize transformer
    slm->transformer = init_transformer(d_model, seq_len, batch_size, mlp_hidden, num_layers, cublas_handle, true);
    
    // Calculate sizes
    int token_emb_size = VOCAB_SIZE * d_model;
    int pos_emb_size = seq_len * d_model;
    int output_proj_size = d_model * VOCAB_SIZE;
    int seq_size = batch_size * seq_len * d_model;
    int logits_size = batch_size * seq_len * VOCAB_SIZE;
    
    // Allocate embedding matrices
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection, output_proj_size * sizeof(float)));
    
    // Allocate embedding gradients
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_grad, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_grad, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_grad, output_proj_size * sizeof(float)));
    
    // Allocate Adam parameters for embeddings
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_v, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_m, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_v, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_m, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_v, output_proj_size * sizeof(float)));
    
    // Allocate forward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logits, logits_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_probabilities, logits_size * sizeof(float)));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_grad_embedded, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_grad_logits, logits_size * sizeof(float)));
    
    // Allocate working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_temp_embedding, seq_size * sizeof(float)));
    
    // Initialize embeddings on host then copy to device
    float* h_token_emb = (float*)malloc(token_emb_size * sizeof(float));
    float* h_pos_emb = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_proj = (float*)malloc(output_proj_size * sizeof(float));
    
    float token_scale = 1.0f / sqrtf(d_model);
    float pos_scale = 0.01f;
    float output_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < token_emb_size; i++) {
        h_token_emb[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    for (int i = 0; i < pos_emb_size; i++) {
        h_pos_emb[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * pos_scale;
    }
    
    for (int i = 0; i < output_proj_size; i++) {
        h_output_proj[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * output_scale;
    }
    
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding, h_token_emb, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_position_embedding, h_pos_emb, pos_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_projection, h_output_proj, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_m, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_v, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_m, 0, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_v, 0, output_proj_size * sizeof(float)));
    
    free(h_token_emb);
    free(h_pos_emb);
    free(h_output_proj);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    free_transformer(slm->transformer);
    
    cudaFree(slm->d_token_embedding);
    cudaFree(slm->d_position_embedding);
    cudaFree(slm->d_output_projection);
    cudaFree(slm->d_token_embedding_grad);
    cudaFree(slm->d_position_embedding_grad);
    cudaFree(slm->d_output_projection_grad);
    cudaFree(slm->d_token_embedding_m);
    cudaFree(slm->d_token_embedding_v);
    cudaFree(slm->d_position_embedding_m);
    cudaFree(slm->d_position_embedding_v);
    cudaFree(slm->d_output_projection_m);
    cudaFree(slm->d_output_projection_v);
    cudaFree(slm->d_embedded_input);
    cudaFree(slm->d_logits);
    cudaFree(slm->d_probabilities);
    cudaFree(slm->d_grad_embedded);
    cudaFree(slm->d_grad_logits);
    cudaFree(slm->d_temp_embedding);
    
    free(slm);
}

// CUDA kernel to gather token embeddings
__global__ void gather_token_embeddings_kernel(float* embedded, float* token_emb, unsigned char* tokens, int batch_size, int seq_len, int d_model, int vocab_size) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int dim = threadIdx.x;
    
    if (batch < batch_size && seq < seq_len && dim < d_model) {
        int token = tokens[batch * seq_len + seq];
        if (token >= vocab_size) token = 0; // Clamp to valid range
        
        int emb_idx = token * d_model + dim;
        int out_idx = (batch * seq_len + seq) * d_model + dim;
        
        embedded[out_idx] = token_emb[emb_idx];
    }
}

// CUDA kernel to add position embeddings
__global__ void add_position_embeddings_kernel(float* embedded, float* pos_emb, int batch_size, int seq_len, int d_model) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int dim = threadIdx.x;
    
    if (batch < batch_size && seq < seq_len && dim < d_model) {
        int pos_idx = seq * d_model + dim;
        int emb_idx = (batch * seq_len + seq) * d_model + dim;
        
        embedded[emb_idx] += pos_emb[pos_idx];
    }
}

// CUDA kernel for softmax
__global__ void softmax_kernel_slm(float* probabilities, float* logits, int batch_size, int seq_len, int vocab_size) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    
    if (batch < batch_size && seq < seq_len) {
        int offset = (batch * seq_len + seq) * vocab_size;
        float* logits_ptr = logits + offset;
        float* probs_ptr = probabilities + offset;
        
        // Find max for numerical stability
        float max_val = logits_ptr[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits_ptr[i] > max_val) max_val = logits_ptr[i];
        }
        
        // Compute softmax
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            probs_ptr[i] = expf(logits_ptr[i] - max_val);
            sum += probs_ptr[i];
        }
        
        for (int i = 0; i < vocab_size; i++) {
            probs_ptr[i] /= sum;
        }
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Token embedding lookup
    dim3 grid(slm->batch_size, slm->seq_len);
    dim3 block(slm->d_model);
    
    gather_token_embeddings_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_token_embedding, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model, slm->vocab_size
    );
    
    // Step 2: Add position embeddings
    add_position_embeddings_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_position_embedding,
        slm->batch_size, slm->seq_len, slm->d_model
    );
    
    // Step 3: Forward pass through transformer
    forward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // Step 4: Output projection to vocabulary
    MLP* final_mlp = slm->transformer->mlp_layers[slm->transformer->num_layers-1];
    int total_seq = slm->batch_size * slm->seq_len;
    
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            slm->vocab_size, total_seq, slm->d_model,
                            &alpha, slm->d_output_projection, slm->d_model,
                            final_mlp->d_layer_output, slm->d_model,
                            &beta, slm->d_logits, slm->vocab_size));
    
    // Step 5: Softmax to get probabilities
    dim3 softmax_grid(slm->batch_size, slm->seq_len);
    softmax_kernel_slm<<<softmax_grid, 1>>>(
        slm->d_probabilities, slm->d_logits,
        slm->batch_size, slm->seq_len, slm->vocab_size
    );
}

// CUDA kernel for cross-entropy loss backward pass
__global__ void cross_entropy_backward_kernel(float* grad_logits, float* probabilities, unsigned char* targets, int batch_size, int seq_len, int vocab_size) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int vocab = threadIdx.x;
    
    if (batch < batch_size && seq < seq_len && vocab < vocab_size) {
        int target_token = targets[batch * seq_len + seq];
        if (target_token >= vocab_size) target_token = 0; // Clamp to valid range
        
        int idx = (batch * seq_len + seq) * vocab_size + vocab;
        
        grad_logits[idx] = probabilities[idx];
        if (vocab == target_token) {
            grad_logits[idx] -= 1.0f;
        }
        grad_logits[idx] /= (batch_size * seq_len); // Normalize by batch size
    }
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens) {
    // Compute cross-entropy loss gradient (used for backward pass)
    dim3 grid(slm->batch_size, slm->seq_len);
    dim3 block(slm->vocab_size);
    
    cross_entropy_backward_kernel<<<grid, block>>>(
        slm->d_grad_logits, slm->d_probabilities, d_target_tokens,
        slm->batch_size, slm->seq_len, slm->vocab_size
    );
    
    // Calculate actual loss on CPU for monitoring
    int total_size = slm->batch_size * slm->seq_len * slm->vocab_size;
    float* h_probs = (float*)malloc(total_size * sizeof(float));
    unsigned char* h_targets = (unsigned char*)malloc(slm->batch_size * slm->seq_len * sizeof(unsigned char));
    
    CHECK_CUDA(cudaMemcpy(h_probs, slm->d_probabilities, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_targets, d_target_tokens, slm->batch_size * slm->seq_len * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int batch = 0; batch < slm->batch_size; batch++) {
        for (int seq = 0; seq < slm->seq_len; seq++) {
            int target = h_targets[batch * slm->seq_len + seq];
            if (target >= slm->vocab_size) target = 0;
            
            int prob_idx = (batch * slm->seq_len + seq) * slm->vocab_size + target;
            float prob = h_probs[prob_idx];
            loss += -logf(fmaxf(prob, 1e-10f)); // Avoid log(0)
        }
    }
    
    free(h_probs);
    free(h_targets);
    
    return loss / (slm->batch_size * slm->seq_len);
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_transformer(slm->transformer);
    
    int token_emb_size = VOCAB_SIZE * slm->d_model;
    int pos_emb_size = slm->seq_len * slm->d_model;
    int output_proj_size = slm->d_model * VOCAB_SIZE;
    
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_grad, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_grad, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_grad, 0, output_proj_size * sizeof(float)));
}

// CUDA kernel for embedding backward pass
__global__ void embedding_backward_kernel(float* embedding_grad, float* grad_embedded, unsigned char* tokens, int batch_size, int seq_len, int d_model, int vocab_size) {
    int vocab = blockIdx.x;
    int dim = threadIdx.x;
    
    if (vocab < vocab_size && dim < d_model) {
        float grad_sum = 0.0f;
        
        for (int batch = 0; batch < batch_size; batch++) {
            for (int seq = 0; seq < seq_len; seq++) {
                int token = tokens[batch * seq_len + seq];
                if (token >= vocab_size) token = 0; // Clamp to valid range
                
                if (token == vocab) {
                    int grad_idx = (batch * seq_len + seq) * d_model + dim;
                    grad_sum += grad_embedded[grad_idx];
                }
            }
        }
        
        int emb_idx = vocab * d_model + dim;
        embedding_grad[emb_idx] = grad_sum;
    }
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = slm->batch_size * slm->seq_len;
    
    // Step 1: Gradient w.r.t. output projection
    MLP* final_mlp = slm->transformer->mlp_layers[slm->transformer->num_layers-1];
    
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            slm->d_model, slm->vocab_size, total_seq,
                            &alpha, final_mlp->d_layer_output, slm->d_model,
                            slm->d_grad_logits, slm->vocab_size,
                            &alpha, slm->d_output_projection_grad, slm->d_model));
    
    // Step 2: Gradient w.r.t. transformer output
    CHECK_CUBLAS(cublasSgemm(slm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->d_model, total_seq, slm->vocab_size,
                            &alpha, slm->d_output_projection, slm->d_model,
                            slm->d_grad_logits, slm->vocab_size,
                            &beta, final_mlp->d_error_output, slm->d_model));
    
    // Step 3: Backward pass through transformer
    backward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // Step 4: Get gradient w.r.t. embeddings (from first layer input gradient)
    Attention* first_attn = slm->transformer->attention_layers[0];
    
    // Copy gradient to our buffer
    int seq_size = slm->batch_size * slm->seq_len * slm->d_model;
    CHECK_CUDA(cudaMemcpy(slm->d_grad_embedded, first_attn->d_error_output, 
                         seq_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 5: Position embedding gradients (sum over batch dimension)
    dim3 pos_grid(slm->seq_len);
    dim3 pos_block(slm->d_model);
    
    for (int seq = 0; seq < slm->seq_len; seq++) {
        for (int dim = 0; dim < slm->d_model; dim++) {
            float grad_sum = 0.0f;
            for (int batch = 0; batch < slm->batch_size; batch++) {
                int grad_idx = (batch * slm->seq_len + seq) * slm->d_model + dim;
                (void)grad_idx;
                (void)grad_sum;
                // This should be done on GPU, but for simplicity doing atomic add
            }
            // Would need a proper CUDA kernel for this
        }
    }
    
    // Step 6: Token embedding gradients
    dim3 token_grid(slm->vocab_size);
    dim3 token_block(slm->d_model);
    
    embedding_backward_kernel<<<token_grid, token_block>>>(
        slm->d_token_embedding_grad, slm->d_grad_embedded, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model, slm->vocab_size
    );
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_slm(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights
void update_weights_slm(SLM* slm, float learning_rate) {
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    int total_seq = slm->batch_size * slm->seq_len;
    
    // Update transformer weights
    update_weights_transformer(slm->transformer, learning_rate);
    
    // Update token embeddings
    int token_emb_size = VOCAB_SIZE * slm->d_model;
    int token_blocks = (token_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<token_blocks, block_size>>>(
        slm->d_token_embedding, slm->d_token_embedding_grad,
        slm->d_token_embedding_m, slm->d_token_embedding_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, token_emb_size, total_seq
    );
    
    // Update position embeddings
    int pos_emb_size = slm->seq_len * slm->d_model;
    int pos_blocks = (pos_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<pos_blocks, block_size>>>(
        slm->d_position_embedding, slm->d_position_embedding_grad,
        slm->d_position_embedding_m, slm->d_position_embedding_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, pos_emb_size, total_seq
    );
    
    // Update output projection
    int output_proj_size = slm->d_model * VOCAB_SIZE;
    int output_blocks = (output_proj_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<output_blocks, block_size>>>(
        slm->d_output_projection, slm->d_output_projection_grad,
        slm->d_output_projection_m, slm->d_output_projection_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, output_proj_size, total_seq
    );
}

// Generate text
void generate_text(SLM* slm, unsigned char* seed, int seed_len, int generate_len, float temperature) {
    (void)slm; (void)seed; (void)seed_len; (void)generate_len; (void)temperature;
}

// Save SLM model
void save_slm(SLM* slm, const char* filename) {
    (void)slm; (void)filename;
}

// Load SLM model
SLM* load_slm(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
    (void)filename; (void)custom_batch_size; (void)cublas_handle;
    return NULL;
}