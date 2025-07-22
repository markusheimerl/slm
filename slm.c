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
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;

    // Initialize 4 SSMs: SSM1 -> SSM2 -> MLP1 -> SSM3 -> SSM4 -> MLP2
    slm->ssm1 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    slm->ssm2 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    slm->ssm3 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    slm->ssm4 = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    
    // Initialize 2 MLPs: first for intermediate processing, second for vocab output
    slm->mlp1 = init_mlp(embed_dim, 4 * embed_dim, embed_dim, seq_len * batch_size);
    slm->mlp2 = init_mlp(embed_dim, 4 * embed_dim, slm->vocab_size, seq_len * batch_size);

    // Allocate embedding matrices
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Allocate working buffers for intermediate outputs
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm2_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_mlp1_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm3_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm4_output, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, seq_len * batch_size * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, seq_len * batch_size * sizeof(float)));
    
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
        if (slm->ssm3) free_ssm(slm->ssm3);
        if (slm->ssm4) free_ssm(slm->ssm4);
        if (slm->mlp1) free_mlp(slm->mlp1);
        if (slm->mlp2) free_mlp(slm->mlp2);
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        cudaFree(slm->d_ssm1_output);
        cudaFree(slm->d_ssm2_output);
        cudaFree(slm->d_mlp1_output);
        cudaFree(slm->d_ssm3_output);
        cudaFree(slm->d_ssm4_output);
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_losses);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->seq_len;
    int batch_size = slm->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // SSM1: Process embedded input
    forward_pass_ssm(slm->ssm1, slm->d_embedded_input);
    // Copy SSM1 output to intermediate buffer
    int ssm_output_size = seq_len * batch_size * slm->embed_dim;
    CHECK_CUDA(cudaMemcpy(slm->d_ssm1_output, slm->ssm1->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // SSM2: Process SSM1 output
    forward_pass_ssm(slm->ssm2, slm->d_ssm1_output);
    // Copy SSM2 output to intermediate buffer
    CHECK_CUDA(cudaMemcpy(slm->d_ssm2_output, slm->ssm2->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // MLP1: Process SSM2 output
    forward_pass_mlp(slm->mlp1, slm->d_ssm2_output);
    // Copy MLP1 output to intermediate buffer
    CHECK_CUDA(cudaMemcpy(slm->d_mlp1_output, slm->mlp1->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // SSM3: Process MLP1 output
    forward_pass_ssm(slm->ssm3, slm->d_mlp1_output);
    // Copy SSM3 output to intermediate buffer
    CHECK_CUDA(cudaMemcpy(slm->d_ssm3_output, slm->ssm3->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // SSM4: Process SSM3 output
    forward_pass_ssm(slm->ssm4, slm->d_ssm3_output);
    // Copy SSM4 output to intermediate buffer
    CHECK_CUDA(cudaMemcpy(slm->d_ssm4_output, slm->ssm4->d_predictions, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // MLP2: Process SSM4 output and produce final vocabulary logits
    forward_pass_mlp(slm->mlp2, slm->d_ssm4_output);
    
    // P_t = softmax(L_t) - Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp2->d_predictions, total_tokens, slm->vocab_size
    );
}
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->seq_len;
    int batch_size = slm->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // ∂L/∂L_t = P_t - 1_{y_t} - Compute cross-entropy gradient (softmax - one_hot) for backprop
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp2->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // L = -log(P_{y_t}) - Calculate actual cross-entropy loss
    cross_entropy_loss_kernel<<<blocks, 256>>>(
        slm->d_losses, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Sum all losses
    float total_loss;
    CHECK_CUBLAS(cublasSasum(slm->ssm1->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &total_loss));
    
    return total_loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_ssm(slm->ssm1);
    zero_gradients_ssm(slm->ssm2);
    zero_gradients_ssm(slm->ssm3);
    zero_gradients_ssm(slm->ssm4);
    zero_gradients_mlp(slm->mlp1);
    zero_gradients_mlp(slm->mlp2);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->seq_len;
    int batch_size = slm->batch_size;
    int ssm_output_size = seq_len * batch_size * slm->embed_dim;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backward through MLP2 (final vocabulary layer)
    backward_pass_mlp(slm->mlp2, slm->d_ssm4_output);
    
    // Copy MLP2 input gradients to SSM4 error
    CHECK_CUDA(cudaMemcpy(slm->ssm4->d_error, slm->mlp2->d_error, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backward through SSM4
    backward_pass_ssm(slm->ssm4, slm->d_ssm3_output);
    
    // Copy SSM4 input gradients to SSM3 error
    CHECK_CUDA(cudaMemset(slm->ssm3->d_error, 0, ssm_output_size * sizeof(float)));
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->ssm3->d_error + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm4->d_state_error + t * batch_size * slm->ssm4->state_dim;
        float* d_output_error_t = slm->ssm4->d_error + t * batch_size * slm->embed_dim;
        
        // ∂L/∂SSM3_out += B^T (∂L/∂SSM4_state) + D^T (∂L/∂SSM4_out)
        CHECK_CUBLAS(cublasSgemm(slm->ssm4->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm4->state_dim,
                                &alpha, slm->ssm4->d_B, slm->embed_dim,
                                d_state_error_t, slm->ssm4->state_dim,
                                &beta, d_input_grad_t, slm->embed_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm4->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->embed_dim,
                                &alpha, slm->ssm4->d_D, slm->embed_dim,
                                d_output_error_t, slm->embed_dim,
                                &alpha, d_input_grad_t, slm->embed_dim));
    }
    
    // Backward through SSM3
    backward_pass_ssm(slm->ssm3, slm->d_mlp1_output);
    
    // Copy SSM3 input gradients to MLP1 error
    CHECK_CUDA(cudaMemset(slm->mlp1->d_error, 0, ssm_output_size * sizeof(float)));
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->mlp1->d_error + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm3->d_state_error + t * batch_size * slm->ssm3->state_dim;
        float* d_output_error_t = slm->ssm3->d_error + t * batch_size * slm->embed_dim;
        
        // ∂L/∂MLP1_out += B^T (∂L/∂SSM3_state) + D^T (∂L/∂SSM3_out)
        CHECK_CUBLAS(cublasSgemm(slm->ssm3->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm3->state_dim,
                                &alpha, slm->ssm3->d_B, slm->embed_dim,
                                d_state_error_t, slm->ssm3->state_dim,
                                &beta, d_input_grad_t, slm->embed_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm3->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->embed_dim,
                                &alpha, slm->ssm3->d_D, slm->embed_dim,
                                d_output_error_t, slm->embed_dim,
                                &alpha, d_input_grad_t, slm->embed_dim));
    }
    
    // Backward through MLP1
    backward_pass_mlp(slm->mlp1, slm->d_ssm2_output);
    
    // Copy MLP1 input gradients to SSM2 error
    CHECK_CUDA(cudaMemcpy(slm->ssm2->d_error, slm->mlp1->d_error, 
                         ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backward through SSM2
    backward_pass_ssm(slm->ssm2, slm->d_ssm1_output);
    
    // Copy SSM2 input gradients to SSM1 error
    CHECK_CUDA(cudaMemset(slm->ssm1->d_error, 0, ssm_output_size * sizeof(float)));
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->ssm1->d_error + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm2->d_state_error + t * batch_size * slm->ssm2->state_dim;
        float* d_output_error_t = slm->ssm2->d_error + t * batch_size * slm->embed_dim;
        
        // ∂L/∂SSM1_out += B^T (∂L/∂SSM2_state) + D^T (∂L/∂SSM2_out)
        CHECK_CUBLAS(cublasSgemm(slm->ssm2->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm2->state_dim,
                                &alpha, slm->ssm2->d_B, slm->embed_dim,
                                d_state_error_t, slm->ssm2->state_dim,
                                &beta, d_input_grad_t, slm->embed_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm2->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->embed_dim,
                                &alpha, slm->ssm2->d_D, slm->embed_dim,
                                d_output_error_t, slm->embed_dim,
                                &alpha, d_input_grad_t, slm->embed_dim));
    }
    
    // Backward through SSM1
    backward_pass_ssm(slm->ssm1, slm->d_embedded_input);
    
    // Compute embedding gradients from SSM1
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         seq_len * batch_size * slm->embed_dim * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm1->d_state_error + t * batch_size * slm->ssm1->state_dim;
        float* d_output_error_t = slm->ssm1->d_error + t * batch_size * slm->embed_dim;
        
        // ∂L/∂E_t = B^T (∂L/∂SSM1_state) + D^T (∂L/∂SSM1_out)
        CHECK_CUBLAS(cublasSgemm(slm->ssm1->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm1->state_dim,
                                &alpha, slm->ssm1->d_B, slm->embed_dim,
                                d_state_error_t, slm->ssm1->state_dim,
                                &beta, d_input_grad_t, slm->embed_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm1->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->embed_dim,
                                &alpha, slm->ssm1->d_D, slm->embed_dim,
                                d_output_error_t, slm->embed_dim,
                                &alpha, d_input_grad_t, slm->embed_dim));
    }
    
    // ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t - Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        batch_size, slm->embed_dim
    );
}

// Update weights using AdamW: W = (1-λη)W - η·m̂/√v̂
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update all SSM weights
    update_weights_ssm(slm->ssm1, learning_rate);
    update_weights_ssm(slm->ssm2, learning_rate);
    update_weights_ssm(slm->ssm3, learning_rate);
    update_weights_ssm(slm->ssm4, learning_rate);
    
    // Update all MLP weights
    update_weights_mlp(slm->mlp1, learning_rate);
    update_weights_mlp(slm->mlp2, learning_rate);
    
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
        embed_size, slm->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Save all SSMs
    char ssm_file[256];
    sprintf(ssm_file, "%s_ssm1.bin", filename);
    save_ssm(slm->ssm1, ssm_file);
    sprintf(ssm_file, "%s_ssm2.bin", filename);
    save_ssm(slm->ssm2, ssm_file);
    sprintf(ssm_file, "%s_ssm3.bin", filename);
    save_ssm(slm->ssm3, ssm_file);
    sprintf(ssm_file, "%s_ssm4.bin", filename);
    save_ssm(slm->ssm4, ssm_file);
    
    // Save all MLPs
    char mlp_file[256];
    sprintf(mlp_file, "%s_mlp1.bin", filename);
    save_mlp(slm->mlp1, mlp_file);
    sprintf(mlp_file, "%s_mlp2.bin", filename);
    save_mlp(slm->mlp2, mlp_file);
    
    // Save embeddings
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* f = fopen(embed_file, "wb");
    if (f) {
        fwrite(&slm->vocab_size, sizeof(int), 1, f);
        fwrite(&slm->embed_dim, sizeof(int), 1, f);
        fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, f);
        fclose(f);
        printf("Embeddings saved to %s\n", embed_file);
    }
    
    free(h_embeddings);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load all SSMs
    char ssm_file[256];
    sprintf(ssm_file, "%s_ssm1.bin", filename);
    SSM* ssm1 = load_ssm(ssm_file, custom_batch_size);
    if (!ssm1) return NULL;
    
    sprintf(ssm_file, "%s_ssm2.bin", filename);
    SSM* ssm2 = load_ssm(ssm_file, custom_batch_size);
    if (!ssm2) { free_ssm(ssm1); return NULL; }
    
    sprintf(ssm_file, "%s_ssm3.bin", filename);
    SSM* ssm3 = load_ssm(ssm_file, custom_batch_size);
    if (!ssm3) { free_ssm(ssm1); free_ssm(ssm2); return NULL; }
    
    sprintf(ssm_file, "%s_ssm4.bin", filename);
    SSM* ssm4 = load_ssm(ssm_file, custom_batch_size);
    if (!ssm4) { free_ssm(ssm1); free_ssm(ssm2); free_ssm(ssm3); return NULL; }
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->ssm1 = ssm1;
    slm->ssm2 = ssm2;
    slm->ssm3 = ssm3;
    slm->ssm4 = ssm4;
    slm->vocab_size = 256; // Fixed vocab size for characters
    slm->embed_dim = ssm1->input_dim;
    slm->seq_len = ssm1->seq_len;
    slm->batch_size = ssm1->batch_size;
    
    // Load all MLPs
    char mlp_file[256];
    sprintf(mlp_file, "%s_mlp1.bin", filename);
    slm->mlp1 = load_mlp(mlp_file, ssm1->seq_len * ssm1->batch_size);
    sprintf(mlp_file, "%s_mlp2.bin", filename);
    slm->mlp2 = load_mlp(mlp_file, ssm1->seq_len * ssm1->batch_size);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm1_output, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm2_output, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_mlp1_output, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm3_output, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_ssm4_output, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, slm->seq_len * slm->batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, slm->seq_len * slm->batch_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, slm->seq_len * slm->batch_size * sizeof(float)));
    
    // Load embeddings
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    FILE* f = fopen(embed_file, "rb");
    if (f) {
        int vocab_size, embed_dim;
        fread(&vocab_size, sizeof(int), 1, f);
        fread(&embed_dim, sizeof(int), 1, f);
        
        float* h_embeddings = (float*)malloc(vocab_size * embed_dim * sizeof(float));
        fread(h_embeddings, sizeof(float), vocab_size * embed_dim, f);
        
        CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                             vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(h_embeddings);
        fclose(f);
        printf("Embeddings loaded from %s\n", embed_file);
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
    
    // Allocate temporary buffers for generation
    unsigned char* h_input = (unsigned char*)malloc(sizeof(unsigned char));
    unsigned char* d_input;
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    float* d_h_current = NULL;
    float* d_h_next = NULL;
    float* d_o_current = NULL;
    float* d_mlp_input = NULL;
    float* d_mlp_hidden = NULL;
    float* d_mlp_output = NULL;
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_h_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h_next, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_o_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_input, slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_hidden, slm->mlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_output, slm->vocab_size * sizeof(float)));
    
    // H_0 = 0 - Initialize hidden state to zero
    CHECK_CUDA(cudaMemset(d_h_current, 0, slm->ssm->state_dim * sizeof(float)));
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text to build up hidden state
    for (int i = 0; i < seed_len; i++) {
        h_input[0] = (unsigned char)seed_text[i];
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // E_t = W_E[X_t] - Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = E_t B^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta, d_h_next, slm->ssm->state_dim));
        
        // H_t += H_{t-1} A^T (except for first character)
        if (i > 0) {
            CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    slm->ssm->state_dim, 1, slm->ssm->state_dim,
                                    &alpha, slm->ssm->d_A, slm->ssm->state_dim,
                                    d_h_current, slm->ssm->state_dim,
                                    &beta_add, d_h_next, slm->ssm->state_dim));
        }
        
        // O_t = H_t σ(H_t)
        int block_size = 256;
        int num_blocks = (slm->ssm->state_dim + block_size - 1) / block_size;
        swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_current, d_h_next, slm->ssm->state_dim);
        
        // Y_t = O_t C^T + E_t D^T
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_C, slm->ssm->state_dim,
                                d_o_current, slm->ssm->state_dim,
                                &beta, d_mlp_input, slm->ssm->output_dim));
        
        // Y_t += E_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta_add, d_mlp_input, slm->ssm->output_dim));
        
        // Z_t = Y_t W_1 - MLP layer 1
        CHECK_CUBLAS(cublasSgemv(slm->mlp->cublas_handle,
                                CUBLAS_OP_T,
                                slm->mlp->input_dim,
                                slm->mlp->hidden_dim,
                                &alpha,
                                slm->mlp->d_fc1_weight,
                                slm->mlp->input_dim,
                                d_mlp_input,
                                1,
                                &beta,
                                d_mlp_hidden,
                                1));

        // A_t = Z_t σ(Z_t) - Apply swish activation
        num_blocks = (slm->mlp->hidden_dim + block_size - 1) / block_size;
        swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
            d_mlp_hidden,
            d_mlp_hidden,
            slm->mlp->hidden_dim
        );

        // L_t = A_t W_2 - MLP layer 2
        CHECK_CUBLAS(cublasSgemv(slm->mlp->cublas_handle,
                                CUBLAS_OP_T,
                                slm->mlp->hidden_dim,
                                slm->mlp->output_dim,
                                &alpha,
                                slm->mlp->d_fc2_weight,
                                slm->mlp->hidden_dim,
                                d_mlp_hidden,
                                1,
                                &beta,
                                d_mlp_output,
                                1));
        
        // Swap current and next
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;
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
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = E_t B^T + H_{t-1} A^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta, d_h_next, slm->ssm->state_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_A, slm->ssm->state_dim,
                                d_h_current, slm->ssm->state_dim,
                                &beta_add, d_h_next, slm->ssm->state_dim));
        
        // O_t = H_t σ(H_t)
        int block_size = 256;
        int num_blocks = (slm->ssm->state_dim + block_size - 1) / block_size;
        swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_current, d_h_next, slm->ssm->state_dim);
        
        // Y_t = O_t C^T + E_t D^T
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_C, slm->ssm->state_dim,
                                d_o_current, slm->ssm->state_dim,
                                &beta, d_mlp_input, slm->ssm->output_dim));
        
        // Y_t += E_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta_add, d_mlp_input, slm->ssm->output_dim));
        
        // Z_t = Y_t W_1 - MLP layer 1
        CHECK_CUBLAS(cublasSgemv(slm->mlp->cublas_handle,
                                CUBLAS_OP_T,
                                slm->mlp->input_dim,
                                slm->mlp->hidden_dim,
                                &alpha,
                                slm->mlp->d_fc1_weight,
                                slm->mlp->input_dim,
                                d_mlp_input,
                                1,
                                &beta,
                                d_mlp_hidden,
                                1));

        // A_t = Z_t σ(Z_t) - Apply swish activation
        num_blocks = (slm->mlp->hidden_dim + block_size - 1) / block_size;
        swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
            d_mlp_hidden,
            d_mlp_hidden,
            slm->mlp->hidden_dim
        );

        // L_t = A_t W_2 - MLP layer 2
        CHECK_CUBLAS(cublasSgemv(slm->mlp->cublas_handle,
                                CUBLAS_OP_T,
                                slm->mlp->hidden_dim,
                                slm->mlp->output_dim,
                                &alpha,
                                slm->mlp->d_fc2_weight,
                                slm->mlp->hidden_dim,
                                d_mlp_hidden,
                                1,
                                &beta,
                                d_mlp_output,
                                1));
        
        // P_t = softmax(L_t) - Apply softmax
        softmax_kernel<<<1, 256>>>(slm->d_softmax, d_mlp_output, 1, slm->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_probs, slm->d_softmax, slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // P_τ(c) = exp(L_c / τ) / Σ_{c'} exp(L_{c'} / τ) - Apply temperature scaling
        if (temperature != 1.0f) {
            float sum = 0.0f;
            for (int j = 0; j < slm->vocab_size; j++) {
                h_probs[j] = expf(logf(h_probs[j] + 1e-15f) / temperature);
                sum += h_probs[j];
            }
            // Renormalize
            for (int j = 0; j < slm->vocab_size; j++) {
                h_probs[j] /= sum;
            }
        }
        
        // Sample from the distribution
        float random_val = (float)rand() / (float)RAND_MAX;
        float cumulative = 0.0f;
        int sampled_char = 0;
        
        for (int j = 0; j < slm->vocab_size; j++) {
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
        
        // Swap current and next
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;
    }
    
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_probs);
    cudaFree(d_input);
    cudaFree(d_h_current);
    cudaFree(d_h_next);
    cudaFree(d_o_current);
    cudaFree(d_mlp_input);
    cudaFree(d_mlp_hidden);
    cudaFree(d_mlp_output);
}