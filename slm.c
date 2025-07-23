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

    // Initialize SSM
    slm->ssm = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    
    // Initialize MLP
    slm->mlp = init_mlp(embed_dim, 4 * embed_dim, slm->vocab_size, seq_len * batch_size);

    // Allocate embedding matrices
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Allocate working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * embed_dim * sizeof(float)));
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
        if (slm->ssm) free_ssm(slm->ssm);
        if (slm->mlp) free_mlp(slm->mlp);
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_losses);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm->seq_len;
    int batch_size = slm->ssm->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // H_t = E_t B^T + H_{t-1} A^T
    // O_t = H_t σ(H_t)  
    // Y_t = O_t C^T + E_t D^T - Forward through SSM
    forward_pass_ssm(slm->ssm, slm->d_embedded_input);

    // Z_t = Y_t W_1
    // A_t = Z_t σ(Z_t)
    // L_t = A_t W_2 - Forward through MLP
    forward_pass_mlp(slm->mlp, slm->ssm->d_predictions);
    
    // P_t = softmax(L_t) - Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm->seq_len;
    int batch_size = slm->ssm->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // ∂L/∂L_t = P_t - 1_{y_t} - Compute cross-entropy gradient (softmax - one_hot) for backprop
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // L = -log(P_{y_t}) - Calculate actual cross-entropy loss
    cross_entropy_loss_kernel<<<blocks, 256>>>(
        slm->d_losses, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Sum all losses
    float total_loss;
    CHECK_CUBLAS(cublasSasum(slm->ssm->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &total_loss));
    
    return total_loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_ssm(slm->ssm);
    zero_gradients_mlp(slm->mlp);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    // ∂L/∂W_2 = A_t^T (∂L/∂L_t)
    // ∂L/∂A_t = (∂L/∂L_t)(W_2)^T
    // ∂L/∂Z_t = ∂L/∂A_t ⊙ [σ(Z_t) + Z_t σ(Z_t)(1-σ(Z_t))]
    // ∂L/∂W_1 = Y_t^T (∂L/∂Z_t) - Backward through MLP
    backward_pass_mlp(slm->mlp, slm->ssm->d_predictions);
    
    // Copy MLP input gradients to SSM
    int total_elements = slm->ssm->seq_len * slm->ssm->batch_size * slm->ssm->output_dim;
    CHECK_CUDA(cudaMemcpy(slm->ssm->d_error, slm->mlp->d_error, 
                         total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // ∂L/∂C = Σ_t (∂L/∂Y_t)^T O_t
    // ∂L/∂D = Σ_t (∂L/∂Y_t)^T E_t  
    // ∂L/∂O_t = (∂L/∂Y_t)C
    // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))] + (∂L/∂H_{t+1})A
    // ∂L/∂A = Σ_t (∂L/∂H_t)^T H_{t-1}
    // ∂L/∂B = Σ_t (∂L/∂H_t)^T E_t - Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_embedded_input);
    
    // ∂L/∂E_t = (∂L/∂H_t) B + (∂L/∂Y_t) D - Compute input gradients from SSM
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         slm->ssm->seq_len * slm->ssm->batch_size * slm->ssm->input_dim * sizeof(float)));
    
    for (int t = 0; t < slm->ssm->seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * slm->ssm->batch_size * slm->ssm->input_dim;
        float* d_state_error_t = slm->ssm->d_state_error + t * slm->ssm->batch_size * slm->ssm->state_dim;
        float* d_output_error_t = slm->ssm->d_error + t * slm->ssm->batch_size * slm->ssm->output_dim;
        
        // ∂L/∂E_t += B^T (∂L/∂H_t) - Gradient from state path
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm->input_dim, slm->ssm->batch_size, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                d_state_error_t, slm->ssm->state_dim,
                                &beta, d_input_grad_t, slm->ssm->input_dim));
        
        // ∂L/∂E_t += D^T (∂L/∂Y_t) - Gradient from output path
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm->input_dim, slm->ssm->batch_size, slm->ssm->output_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                d_output_error_t, slm->ssm->output_dim,
                                &alpha, d_input_grad_t, slm->ssm->input_dim));
    }
    
    // ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t - Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((slm->ssm->batch_size + 255) / 256, slm->ssm->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        slm->ssm->batch_size, slm->embed_dim
    );
}

// Update weights using AdamW: W = (1-λη)W - η·m̂/√v̂
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
    
    // Update MLP weights
    update_weights_mlp(slm->mlp, learning_rate);
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssm->beta1, slm->ssm->t);
    float beta2_t = powf(slm->ssm->beta2, slm->ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssm->beta1, slm->ssm->beta2, slm->ssm->epsilon,
        learning_rate, slm->ssm->weight_decay, alpha_t,
        embed_size, slm->ssm->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    save_ssm(slm->ssm, filename);
    
    // Save MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    char* dot = strrchr(mlp_file, '.');
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
        fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, f);
        fclose(f);
        printf("Embeddings saved to %s\n", embed_file);
    }
    
    free(h_embeddings);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    SSM* ssm = load_ssm(filename, custom_batch_size);
    if (!ssm) return NULL;
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->ssm = ssm;
    slm->vocab_size = ssm->output_dim;
    slm->embed_dim = ssm->input_dim;
    
    // Load MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    char* dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    slm->mlp = load_mlp(mlp_file, ssm->seq_len * ssm->batch_size);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, ssm->seq_len * ssm->batch_size * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, ssm->seq_len * ssm->batch_size * ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, ssm->seq_len * ssm->batch_size * ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, ssm->seq_len * ssm->batch_size * sizeof(float)));
    
    // Load embeddings
    char embed_file[256];
    strcpy(embed_file, filename);
    dot = strrchr(embed_file, '.');
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
    
    // Store original dimensions to restore later
    int original_seq_len = slm->ssm->seq_len;
    int original_ssm_batch_size = slm->ssm->batch_size;
    int original_mlp_batch_size = slm->mlp->batch_size;
    
    // Temporarily set dimensions for single-token processing
    slm->ssm->seq_len = 1;
    slm->ssm->batch_size = 1;
    slm->mlp->batch_size = 1;
    
    // Allocate temporary buffers for generation
    unsigned char* h_input = (unsigned char*)malloc(sizeof(unsigned char));
    unsigned char* d_input;
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    CHECK_CUDA(cudaMemset(slm->ssm->d_states, 0, slm->ssm->state_dim * sizeof(float)));
    
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
        
        // Forward through SSM
        forward_pass_ssm(slm->ssm, slm->d_embedded_input);
        
        // No need for MLP during seed processing - only building SSM state
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
        
        // Forward through SSM
        forward_pass_ssm(slm->ssm, slm->d_embedded_input);
        
        // Forward through MLP
        forward_pass_mlp(slm->mlp, slm->ssm->d_predictions);
        
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 256>>>(
            slm->d_softmax, slm->mlp->d_predictions, 1, slm->vocab_size
        );
        
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
    }
    
    printf("\n");
    
    // Restore original dimensions
    slm->ssm->seq_len = original_seq_len;
    slm->ssm->batch_size = original_ssm_batch_size;
    slm->mlp->batch_size = original_mlp_batch_size;
    
    // Cleanup
    free(h_input);
    free(h_probs);
    cudaFree(d_input);
}