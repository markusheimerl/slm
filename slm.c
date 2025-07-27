#include "slm.h"

// CUDA kernel for embedding lookup: E_t = W_E[X_t]
__global__ void embedding_lookup_kernel(float* output, float* embeddings, unsigned char* chars, 
                                       int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockIdx.y * batch_size;
    
    if (idx < batch_size) {
        int char_idx = chars[total + idx];
        for (int i = 0; i < embed_dim; i++) {
            output[(total + idx) * embed_dim + i] = embeddings[char_idx * embed_dim + i];
        }
    }
}

// CUDA kernel for softmax: P_t = exp(L_t) / Σ_c exp(L_{t,c})
__global__ void softmax_kernel(float* output, float* input, int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* in = input + idx * vocab_size;
    float* out = output + idx * vocab_size;
    
    // Find max for numerical stability
    float max_val = in[0];
    for (int i = 1; i < vocab_size; i++) {
        max_val = fmaxf(max_val, in[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        out[i] /= sum;
    }
}

// CUDA kernel for cross-entropy gradient: ∂L/∂L_t = P_t - 1_{y_t}
__global__ void cross_entropy_gradient_kernel(float* grad, float* softmax, unsigned char* targets, 
                                             int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* grad_ptr = grad + idx * vocab_size;
    float* softmax_ptr = softmax + idx * vocab_size;
    int target = targets[idx];
    
    // Gradient is softmax - one_hot(target)
    for (int i = 0; i < vocab_size; i++) {
        grad_ptr[i] = softmax_ptr[i];
    }
    grad_ptr[target] -= 1.0f;
}

// CUDA kernel for cross-entropy loss: L = -log(P_{y_t})
__global__ void cross_entropy_loss_kernel(float* losses, float* softmax, unsigned char* targets, 
                                         int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* softmax_ptr = softmax + idx * vocab_size;
    int target = targets[idx];
    
    // Loss = -log(softmax[target])
    float prob = fmaxf(softmax_ptr[target], 1e-15f); // Avoid log(0)
    losses[idx] = -logf(prob);
}

// CUDA kernel for KL divergence loss: KL(P||U) = Σ P(c) log(P(c)) + log(V)
// Since log(V) is constant, we compute only the entropy: -Σ P(c) log(P(c))
__global__ void kl_divergence_loss_kernel(float* kl_losses, float* softmax, 
                                        int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* softmax_ptr = softmax + idx * vocab_size;
    
    // Compute negative entropy: -Σ P(c) log(P(c))
    float neg_entropy = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float prob = fmaxf(softmax_ptr[i], 1e-15f); // Avoid log(0)
        neg_entropy -= prob * logf(prob);
    }
    kl_losses[idx] = neg_entropy;
}

// CUDA kernel for KL divergence gradient: ∂KL/∂L_t = kl_weight * P(c) * (log(P(c)) + 1 - log(V))
// For uniform reference distribution U, log(V) is constant, so gradient is: kl_weight * P(c) * (log(P(c)) + 1)
__global__ void kl_divergence_gradient_kernel(float* grad, float* softmax, float kl_weight,
                                            int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* grad_ptr = grad + idx * vocab_size;
    float* softmax_ptr = softmax + idx * vocab_size;
    
    // Add KL divergence gradient to existing gradient
    for (int i = 0; i < vocab_size; i++) {
        float prob = fmaxf(softmax_ptr[i], 1e-15f); // Avoid log(0)
        grad_ptr[i] += kl_weight * prob * (logf(prob) + 1.0f);
    }
}

// CUDA kernel for embedding gradient accumulation: ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t
__global__ void embedding_gradient_kernel(float* embed_grad, float* input_grad, unsigned char* chars,
                                         int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blockIdx.y * batch_size;
    
    if (idx < batch_size) {
        int char_idx = chars[total + idx];
        for (int i = 0; i < embed_dim; i++) {
            atomicAdd(&embed_grad[char_idx * embed_dim + i], 
                     input_grad[(total + idx) * embed_dim + i]);
        }
    }
}

// Initialize SLM
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
        
    // Set dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    
    // Set KL divergence penalty weight (can be tuned)
    slm->kl_penalty_weight = 0.01f;

    // Initialize first SSM (embed_dim -> embed_dim)
    slm->ssm1 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    
    // Initialize second SSM (embed_dim -> embed_dim)  
    slm->ssm2 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    
    // Initialize MLP
    slm->mlp = init_mlp(embed_dim, 4 * embed_dim, slm->vocab_size, seq_len * batch_size);

    // Allocate embedding matrices
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Allocate working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_gradients, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, seq_len * batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_kl_losses, seq_len * batch_size * sizeof(float)));
    
    // Initialize embeddings with Xavier initialization
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    float scale = sqrtf(2.0f / slm->embed_dim);
    for (int i = 0; i < slm->vocab_size * slm->embed_dim; i++) {
        h_embeddings[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_m, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_v, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    free(h_embeddings);
    return slm;
}

// Free SLM
void free_slm(SLM* slm) {
    if (slm) {
        if (slm->ssm1) free_ssm(slm->ssm1);
        if (slm->ssm2) free_ssm(slm->ssm2);
        if (slm->mlp) free_mlp(slm->mlp);
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        cudaFree(slm->d_ssm1_output);
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_ssm1_gradients);
        cudaFree(slm->d_losses);
        cudaFree(slm->d_kl_losses);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm1->seq_len;
    int batch_size = slm->ssm1->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // First SSM layer: Embedding -> SSM1
    // H1_t = E_t B1^T + H1_{t-1} A1^T
    // O1_t = H1_t σ(H1_t)  
    // Y1_t = O1_t C1^T + E_t D1^T
    reset_state_ssm(slm->ssm1);
    for (int t = 0; t < seq_len; t++) {
        float* d_embedded_t = slm->d_embedded_input + t * batch_size * slm->embed_dim;
        forward_pass_ssm(slm->ssm1, d_embedded_t, t);
    }
    
    // Copy SSM1 output to intermediate buffer
    int ssm_output_size = seq_len * batch_size * slm->embed_dim;
    CHECK_CUDA(cudaMemcpy(slm->d_ssm1_output, slm->ssm1->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Second SSM layer: SSM1 output -> SSM2
    // H2_t = Y1_t B2^T + H2_{t-1} A2^T
    // O2_t = H2_t σ(H2_t)
    // Y2_t = O2_t C2^T + Y1_t D2^T
    reset_state_ssm(slm->ssm2);
    for (int t = 0; t < seq_len; t++) {
        float* d_ssm1_output_t = slm->d_ssm1_output + t * batch_size * slm->embed_dim;
        forward_pass_ssm(slm->ssm2, d_ssm1_output_t, t);
    }

    // Z_t = Y2_t W_1
    // A_t = Z_t σ(Z_t)
    // L_t = A_t W_2 - Forward through MLP
    forward_pass_mlp(slm->mlp, slm->ssm2->d_predictions);
    
    // P_t = softmax(L_t) - Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}} + λ_KL * KL(P||U)
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm1->seq_len;
    int batch_size = slm->ssm1->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // ∂L/∂L_t = P_t - 1_{y_t} - Compute cross-entropy gradient (softmax - one_hot) for backprop
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Add KL divergence gradient to the existing gradient
    if (slm->kl_penalty_weight > 0.0f) {
        kl_divergence_gradient_kernel<<<blocks, 256>>>(
            slm->mlp->d_error, slm->d_softmax, slm->kl_penalty_weight, total_tokens, slm->vocab_size
        );
    }
    
    // L_CE = -log(P_{y_t}) - Calculate actual cross-entropy loss
    cross_entropy_loss_kernel<<<blocks, 256>>>(
        slm->d_losses, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Calculate KL divergence loss if penalty weight > 0
    float kl_loss = 0.0f;
    if (slm->kl_penalty_weight > 0.0f) {
        kl_divergence_loss_kernel<<<blocks, 256>>>(
            slm->d_kl_losses, slm->d_softmax, total_tokens, slm->vocab_size
        );
        
        // Sum KL divergence losses
        CHECK_CUBLAS(cublasSasum(slm->ssm1->cublas_handle, total_tokens, 
                                slm->d_kl_losses, 1, &kl_loss));
        kl_loss /= total_tokens;
    }
    
    // Sum cross-entropy losses
    float ce_loss;
    CHECK_CUBLAS(cublasSasum(slm->ssm1->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &ce_loss));
    ce_loss /= total_tokens;
    
    // Total loss = cross-entropy + KL penalty
    return ce_loss + slm->kl_penalty_weight * kl_loss;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_ssm(slm->ssm1);
    zero_gradients_ssm(slm->ssm2);
    zero_gradients_mlp(slm->mlp);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    // ∂L/∂W_2 = A_t^T (∂L/∂L_t)
    // ∂L/∂A_t = (∂L/∂L_t)(W_2)^T
    // ∂L/∂Z_t = ∂L/∂A_t ⊙ [σ(Z_t) + Z_t σ(Z_t)(1-σ(Z_t))]
    // ∂L/∂W_1 = Y2_t^T (∂L/∂Z_t) - Backward through MLP
    backward_pass_mlp(slm->mlp, slm->ssm2->d_predictions);
    
    // Copy MLP input gradients to second SSM output error
    int total_elements = slm->ssm2->seq_len * slm->ssm2->batch_size * slm->ssm2->output_dim;
    CHECK_CUDA(cudaMemcpy(slm->ssm2->d_error, slm->mlp->d_error, 
                         total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backward through second SSM layer (SSM2)
    // ∂L/∂C2 = Σ_t (∂L/∂Y2_t)^T O2_t
    // ∂L/∂D2 = Σ_t (∂L/∂Y2_t)^T Y1_t  
    // ∂L/∂O2_t = (∂L/∂Y2_t)C2
    // ∂L/∂H2_t = ∂L/∂O2_t ⊙ [σ(H2_t) + H2_t σ(H2_t)(1-σ(H2_t))] + (∂L/∂H2_{t+1})A2
    // ∂L/∂A2 = Σ_t (∂L/∂H2_t)^T H2_{t-1}
    // ∂L/∂B2 = Σ_t (∂L/∂H2_t)^T Y1_t
    backward_pass_ssm(slm->ssm2, slm->d_ssm1_output);
    
    // Compute gradients with respect to SSM1 output (input to SSM2)
    // ∂L/∂Y1_t = (∂L/∂H2_t) B2 + (∂L/∂Y2_t) D2
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CHECK_CUDA(cudaMemset(slm->d_ssm1_gradients, 0, 
                         slm->ssm2->seq_len * slm->ssm2->batch_size * slm->ssm2->input_dim * sizeof(float)));
    
    for (int t = 0; t < slm->ssm2->seq_len; t++) {
        float* d_ssm1_grad_t = slm->d_ssm1_gradients + t * slm->ssm2->batch_size * slm->ssm2->input_dim;
        float* d_state_error_t = slm->ssm2->d_state_error + t * slm->ssm2->batch_size * slm->ssm2->state_dim;
        float* d_output_error_t = slm->ssm2->d_error + t * slm->ssm2->batch_size * slm->ssm2->output_dim;
        
        // ∂L/∂Y1_t += B2^T (∂L/∂H2_t) - Gradient from state path
        CHECK_CUBLAS(cublasSgemm(slm->ssm2->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm2->input_dim, slm->ssm2->batch_size, slm->ssm2->state_dim,
                                &alpha, slm->ssm2->d_B, slm->ssm2->input_dim,
                                d_state_error_t, slm->ssm2->state_dim,
                                &beta, d_ssm1_grad_t, slm->ssm2->input_dim));
        
        // ∂L/∂Y1_t += D2^T (∂L/∂Y2_t) - Gradient from output path
        CHECK_CUBLAS(cublasSgemm(slm->ssm2->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm2->input_dim, slm->ssm2->batch_size, slm->ssm2->output_dim,
                                &alpha, slm->ssm2->d_D, slm->ssm2->input_dim,
                                d_output_error_t, slm->ssm2->output_dim,
                                &alpha, d_ssm1_grad_t, slm->ssm2->input_dim));
    }
    
    // Copy gradients to first SSM output error
    CHECK_CUDA(cudaMemcpy(slm->ssm1->d_error, slm->d_ssm1_gradients, 
                         total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backward through first SSM layer (SSM1)
    // ∂L/∂C1 = Σ_t (∂L/∂Y1_t)^T O1_t
    // ∂L/∂D1 = Σ_t (∂L/∂Y1_t)^T E_t  
    // ∂L/∂O1_t = (∂L/∂Y1_t)C1
    // ∂L/∂H1_t = ∂L/∂O1_t ⊙ [σ(H1_t) + H1_t σ(H1_t)(1-σ(H1_t))] + (∂L/∂H1_{t+1})A1
    // ∂L/∂A1 = Σ_t (∂L/∂H1_t)^T H1_{t-1}
    // ∂L/∂B1 = Σ_t (∂L/∂H1_t)^T E_t
    backward_pass_ssm(slm->ssm1, slm->d_embedded_input);
    
    // ∂L/∂E_t = (∂L/∂H1_t) B1 + (∂L/∂Y1_t) D1 - Compute input gradients from first SSM
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         slm->ssm1->seq_len * slm->ssm1->batch_size * slm->ssm1->input_dim * sizeof(float)));
    
    for (int t = 0; t < slm->ssm1->seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * slm->ssm1->batch_size * slm->ssm1->input_dim;
        float* d_state_error_t = slm->ssm1->d_state_error + t * slm->ssm1->batch_size * slm->ssm1->state_dim;
        float* d_output_error_t = slm->ssm1->d_error + t * slm->ssm1->batch_size * slm->ssm1->output_dim;
        
        // ∂L/∂E_t += B1^T (∂L/∂H1_t) - Gradient from state path
        CHECK_CUBLAS(cublasSgemm(slm->ssm1->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm1->input_dim, slm->ssm1->batch_size, slm->ssm1->state_dim,
                                &alpha, slm->ssm1->d_B, slm->ssm1->input_dim,
                                d_state_error_t, slm->ssm1->state_dim,
                                &beta, d_input_grad_t, slm->ssm1->input_dim));
        
        // ∂L/∂E_t += D1^T (∂L/∂Y1_t) - Gradient from output path
        CHECK_CUBLAS(cublasSgemm(slm->ssm1->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm1->input_dim, slm->ssm1->batch_size, slm->ssm1->output_dim,
                                &alpha, slm->ssm1->d_D, slm->ssm1->input_dim,
                                d_output_error_t, slm->ssm1->output_dim,
                                &alpha, d_input_grad_t, slm->ssm1->input_dim));
    }
    
    // ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t - Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((slm->ssm1->batch_size + 255) / 256, slm->ssm1->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        slm->ssm1->batch_size, slm->embed_dim
    );
}

// Update weights using AdamW: W = (1-λη)W - η·m̂/√v̂
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update both SSM weights
    update_weights_ssm(slm->ssm1, learning_rate);
    update_weights_ssm(slm->ssm2, learning_rate);
    
    // Update MLP weights
    update_weights_mlp(slm->mlp, learning_rate);
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssm1->beta1, slm->ssm1->t);
    float beta2_t = powf(slm->ssm1->beta2, slm->ssm1->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssm1->beta1, slm->ssm1->beta2, slm->ssm1->epsilon,
        learning_rate, slm->ssm1->weight_decay, alpha_t,
        embed_size, slm->ssm1->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Save first SSM
    save_ssm(slm->ssm1, filename);
    
    // Save second SSM  
    char ssm2_file[256];
    strcpy(ssm2_file, filename);
    char* dot = strrchr(ssm2_file, '.');
    if (dot) *dot = '\0';
    strcat(ssm2_file, "_ssm2.bin");
    save_ssm(slm->ssm2, ssm2_file);
    
    // Save MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    save_mlp(slm->mlp, mlp_file);
    
    // Save embeddings
    char embed_file[256];
    strcpy(embed_file, filename);
    dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* f = fopen(embed_file, "wb");
    if (f) {
        fwrite(&slm->vocab_size, sizeof(int), 1, f);
        fwrite(&slm->embed_dim, sizeof(int), 1, f);
        fwrite(&slm->kl_penalty_weight, sizeof(float), 1, f);
        fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, f);
        fclose(f);
        printf("Embeddings saved to %s\n", embed_file);
    }
    
    free(h_embeddings);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load first SSM
    SSM* ssm1 = load_ssm(filename, custom_batch_size);
    if (!ssm1) return NULL;
    
    // Load second SSM
    char ssm2_file[256];
    strcpy(ssm2_file, filename);
    char* dot = strrchr(ssm2_file, '.');
    if (dot) *dot = '\0';
    strcat(ssm2_file, "_ssm2.bin");
    SSM* ssm2 = load_ssm(ssm2_file, custom_batch_size);
    if (!ssm2) {
        free_ssm(ssm1);
        return NULL;
    }
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->ssm1 = ssm1;
    slm->ssm2 = ssm2;
    slm->vocab_size = ssm1->output_dim;
    slm->embed_dim = ssm1->input_dim;
    
    // Set default KL divergence penalty weight
    slm->kl_penalty_weight = 0.01f;
    
    // Load MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    slm->mlp = load_mlp(mlp_file, ssm1->seq_len * ssm1->batch_size);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, ssm1->seq_len * ssm1->batch_size * ssm1->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_output, ssm1->seq_len * ssm1->batch_size * ssm1->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, ssm1->seq_len * ssm1->batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, ssm1->seq_len * ssm1->batch_size * ssm1->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_gradients, ssm1->seq_len * ssm1->batch_size * ssm1->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, ssm1->seq_len * ssm1->batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_kl_losses, ssm1->seq_len * ssm1->batch_size * sizeof(float)));
    
    // Load embeddings
    char embed_file[256];
    strcpy(embed_file, filename);
    dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    FILE* f = fopen(embed_file, "rb");
    if (f) {
        int vocab_size, embed_dim;
        float kl_penalty_weight;
        fread(&vocab_size, sizeof(int), 1, f);
        fread(&embed_dim, sizeof(int), 1, f);
        fread(&kl_penalty_weight, sizeof(float), 1, f);
        
        slm->kl_penalty_weight = kl_penalty_weight;
        
        float* h_embeddings = (float*)malloc(vocab_size * embed_dim * sizeof(float));
        fread(h_embeddings, sizeof(float), vocab_size * embed_dim, f);
        
        CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                             vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(h_embeddings);
        fclose(f);
        printf("Embeddings loaded from %s (KL penalty weight: %.4f)\n", embed_file, kl_penalty_weight);
    }
    
    CHECK_CUDA(cudaMemset(slm->d_embeddings_m, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_v, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    return slm;
}

// Text generation function
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature) {
    int seed_len = strlen(seed_text);
    if (seed_len == 0) {
        printf("Error: Empty seed text\n");
        return;
    }
    
    // Create a temporary SLM instance for generation with batch_size=1
    int max_timesteps = seed_len + generation_length;
    SLM* gen_slm = init_slm(slm->embed_dim, slm->ssm1->state_dim, max_timesteps, 1);
    
    // Copy trained weights from main model to generation model
    // Copy first SSM weights
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm1->d_A, slm->ssm1->d_A, 
                         slm->ssm1->state_dim * slm->ssm1->state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm1->d_B, slm->ssm1->d_B, 
                         slm->ssm1->state_dim * slm->ssm1->input_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm1->d_C, slm->ssm1->d_C, 
                         slm->ssm1->output_dim * slm->ssm1->state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm1->d_D, slm->ssm1->d_D, 
                         slm->ssm1->output_dim * slm->ssm1->input_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Copy second SSM weights
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm2->d_A, slm->ssm2->d_A, 
                         slm->ssm2->state_dim * slm->ssm2->state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm2->d_B, slm->ssm2->d_B, 
                         slm->ssm2->state_dim * slm->ssm2->input_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm2->d_C, slm->ssm2->d_C, 
                         slm->ssm2->output_dim * slm->ssm2->state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->ssm2->d_D, slm->ssm2->d_D, 
                         slm->ssm2->output_dim * slm->ssm2->input_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Copy MLP weights
    CHECK_CUDA(cudaMemcpy(gen_slm->mlp->d_W1, slm->mlp->d_W1,
                         slm->mlp->hidden_dim * slm->mlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_slm->mlp->d_W2, slm->mlp->d_W2,
                         slm->mlp->output_dim * slm->mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    // Copy embeddings
    CHECK_CUDA(cudaMemcpy(gen_slm->d_embeddings, slm->d_embeddings,
                         slm->vocab_size * slm->embed_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    // Allocate temporary buffers for generation
    unsigned char* h_input = (unsigned char*)malloc(sizeof(unsigned char));
    unsigned char* d_input;
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    
    // Reset SSM state for generation
    reset_state_ssm(gen_slm->ssm1);
    reset_state_ssm(gen_slm->ssm2);
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text to build up hidden state
    for (int i = 0; i < seed_len; i++) {
        h_input[0] = (unsigned char)seed_text[i];
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // E_t = W_E[X_t] - Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            gen_slm->d_embedded_input, gen_slm->d_embeddings, d_input, 1, gen_slm->embed_dim
        );
        
        // Forward through first SSM
        forward_pass_ssm(gen_slm->ssm1, gen_slm->d_embedded_input, i);
        
        // Get the first SSM output for this timestep and copy to intermediate buffer
        float* d_y1_t = gen_slm->ssm1->d_predictions + i * 1 * gen_slm->ssm1->output_dim;
        float* d_ssm1_out_t = gen_slm->d_ssm1_output + i * 1 * gen_slm->embed_dim;
        CHECK_CUDA(cudaMemcpy(d_ssm1_out_t, d_y1_t, 
                             gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Forward through second SSM
        forward_pass_ssm(gen_slm->ssm2, d_ssm1_out_t, i);
    }
    
    // Now generate new characters
    for (int i = 0; i < generation_length; i++) {
        // Use the last character of seed (if first generation) or the last generated character
        if (i == 0) {
            h_input[0] = (unsigned char)seed_text[seed_len - 1];
        }
        
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // E_t = W_E[X_t] - Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            gen_slm->d_embedded_input, gen_slm->d_embeddings, d_input, 1, gen_slm->embed_dim
        );
        
        // Forward through first SSM
        int timestep = seed_len + i;
        forward_pass_ssm(gen_slm->ssm1, gen_slm->d_embedded_input, timestep);
        
        // Get the first SSM output for this timestep and copy to intermediate buffer
        float* d_y1_t = gen_slm->ssm1->d_predictions + timestep * 1 * gen_slm->ssm1->output_dim;
        float* d_ssm1_out_t = gen_slm->d_ssm1_output + timestep * 1 * gen_slm->embed_dim;
        CHECK_CUDA(cudaMemcpy(d_ssm1_out_t, d_y1_t, 
                             gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Forward through second SSM
        forward_pass_ssm(gen_slm->ssm2, d_ssm1_out_t, timestep);
        
        // Get the second SSM output for this timestep
        float* d_y2_t = gen_slm->ssm2->d_predictions + timestep * 1 * gen_slm->ssm2->output_dim;
        
        // Forward through MLP
        forward_pass_mlp(gen_slm->mlp, d_y2_t);
        
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 256>>>(
            gen_slm->d_softmax, gen_slm->mlp->d_predictions, 1, gen_slm->vocab_size
        );
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_probs, gen_slm->d_softmax, gen_slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // P_τ(c) = exp(L_c / τ) / Σ_{c'} exp(L_{c'} / τ) - Apply temperature scaling
        if (temperature != 1.0f) {
            float sum = 0.0f;
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                h_probs[j] = expf(logf(h_probs[j] + 1e-15f) / temperature);
                sum += h_probs[j];
            }
            // Renormalize
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                h_probs[j] /= sum;
            }
        }
        
        // Sample from the distribution
        float random_val = (float)rand() / (float)RAND_MAX;
        float cumulative = 0.0f;
        int sampled_char = 0;
        
        for (int j = 0; j < gen_slm->vocab_size; j++) {
            cumulative += h_probs[j];
            if (random_val <= cumulative) {
                sampled_char = j;
                break;
            }
        }
        
        // Ensure we have a valid printable character
        if (sampled_char < 32 || sampled_char > 126) {
            sampled_char = 32; // space
        }
        
        printf("%c", sampled_char);
        fflush(stdout);
        
        // Set up for next iteration
        h_input[0] = (unsigned char)sampled_char;
    }
    
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_probs);
    cudaFree(d_input);
    free_slm(gen_slm);
}

// Set KL divergence penalty weight
void set_kl_penalty_weight_slm(SLM* slm, float weight) {
    if (slm) {
        slm->kl_penalty_weight = weight;
        printf("KL divergence penalty weight set to: %.4f\n", weight);
    }
}