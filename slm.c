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
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int num_layers, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
        
    // Set dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    slm->num_layers = num_layers;

    // Initialize SSM layers (embed_dim -> embed_dim)
    slm->ssms = (SSM**)malloc(num_layers * sizeof(SSM*));
    for (int i = 0; i < num_layers; i++) {
        slm->ssms[i] = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    }
    
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
    
    // Allocate arrays for intermediate SSM outputs and gradients
    if (num_layers > 1) {
        slm->d_ssm_outputs = (float**)malloc((num_layers - 1) * sizeof(float*));
        slm->d_ssm_gradients = (float**)malloc((num_layers - 1) * sizeof(float*));
        
        for (int i = 0; i < num_layers - 1; i++) {
            CHECK_CUDA(cudaMalloc(&slm->d_ssm_outputs[i], seq_len * batch_size * embed_dim * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&slm->d_ssm_gradients[i], seq_len * batch_size * embed_dim * sizeof(float)));
        }
    } else {
        slm->d_ssm_outputs = NULL;
        slm->d_ssm_gradients = NULL;
    }
    
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
        for (int i = 0; i < slm->num_layers; i++) {
            if (slm->ssms[i]) free_ssm(slm->ssms[i]);
        }
        free(slm->ssms);
        
        if (slm->mlp) free_mlp(slm->mlp);
        
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_losses);
        
        if (slm->num_layers > 1) {
            for (int i = 0; i < slm->num_layers - 1; i++) {
                cudaFree(slm->d_ssm_outputs[i]);
                cudaFree(slm->d_ssm_gradients[i]);
            }
            free(slm->d_ssm_outputs);
            free(slm->d_ssm_gradients);
        }
        
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssms[0]->seq_len;
    int batch_size = slm->ssms[0]->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // Process through all SSM layers in sequence
    float* current_input = slm->d_embedded_input;
    
    for (int layer = 0; layer < slm->num_layers; layer++) {
        // Reset SSM state for this layer
        reset_state_ssm(slm->ssms[layer]);
        
        // Forward through timesteps for current layer
        for (int t = 0; t < seq_len; t++) {
            float* input_t = current_input + t * batch_size * slm->embed_dim;
            // H_t = X_t B^T + H_{t-1} A^T
            // O_t = H_t σ(H_t)  
            // Y_t = O_t C^T + X_t D^T
            forward_pass_ssm(slm->ssms[layer], input_t, t);
        }
        
        // For all but the last layer, copy output to intermediate buffer and set as next input
        if (layer < slm->num_layers - 1) {
            int output_size = seq_len * batch_size * slm->embed_dim;
            CHECK_CUDA(cudaMemcpy(slm->d_ssm_outputs[layer], slm->ssms[layer]->d_predictions, 
                                 output_size * sizeof(float), cudaMemcpyDeviceToDevice));
            current_input = slm->d_ssm_outputs[layer];
        }
    }

    // Z_t = Y_last_t W_1
    // A_t = Z_t σ(Z_t)
    // L_t = A_t W_2 - Forward through MLP
    forward_pass_mlp(slm->mlp, slm->ssms[slm->num_layers - 1]->d_predictions);
    
    // P_t = softmax(L_t) - Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssms[0]->seq_len;
    int batch_size = slm->ssms[0]->batch_size;
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
    CHECK_CUBLAS(cublasSasum(slm->ssms[0]->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &total_loss));
    
    return total_loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    for (int i = 0; i < slm->num_layers; i++) {
        zero_gradients_ssm(slm->ssms[i]);
    }
    zero_gradients_mlp(slm->mlp);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backward through MLP: Compute gradients for MLP weights and hidden layer errors
    // ∂L/∂W_2 = A_t^T (∂L/∂L_t) - MLP output weight gradients
    // ∂L/∂A_t = (∂L/∂L_t)(W_2)^T - Gradients w.r.t. MLP hidden activations
    // ∂L/∂Z_t = ∂L/∂A_t ⊙ [σ(Z_t) + Z_t σ(Z_t)(1-σ(Z_t))] - Swish derivative through hidden layer
    // ∂L/∂W_1 = Y_last_t^T (∂L/∂Z_t) - MLP input weight gradients
    backward_pass_mlp(slm->mlp, slm->ssms[slm->num_layers - 1]->d_predictions);
    
    // Compute gradients w.r.t. MLP input (which flows back to last SSM output):
    // ∂L/∂Y_last = ∂L/∂Z * W1^T - Chain rule from MLP hidden errors to last SSM output
    // This computes: [seq_len*batch_size, embed_dim] = [seq_len*batch_size, hidden_dim] * [hidden_dim, embed_dim]^T
    CHECK_CUBLAS(cublasSgemm(slm->mlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->mlp->input_dim, slm->mlp->batch_size, slm->mlp->hidden_dim,
                            &alpha, slm->mlp->d_W1, slm->mlp->input_dim,
                            slm->mlp->d_error_hidden, slm->mlp->hidden_dim,
                            &beta, slm->ssms[slm->num_layers - 1]->d_error, slm->mlp->input_dim));
    
    // Process SSM layers in reverse order (last → ... → 1 → 0)
    for (int layer = slm->num_layers - 1; layer >= 0; layer--) {
        // Get the input that was used for this layer during forward pass
        float* layer_input = (layer == 0) ? slm->d_embedded_input : slm->d_ssm_outputs[layer - 1];
        
        // Backward pass through current SSM layer:
        // ∂L/∂C = Σ_t (∂L/∂Y_t)^T O_t - Output projection weight gradients
        // ∂L/∂D = Σ_t (∂L/∂Y_t)^T X_t - Feedthrough weight gradients
        // ∂L/∂O_t = (∂L/∂Y_t)C - Gradients w.r.t. activated states
        // ∂L/∂H_t = ∂L/∂O_t ⊙ [σ(H_t) + H_t σ(H_t)(1-σ(H_t))] + (∂L/∂H_{t+1})A - Swish derivative + temporal flow
        // ∂L/∂A = Σ_t (∂L/∂H_t)^T H_{t-1} - State transition weight gradients
        // ∂L/∂B = Σ_t (∂L/∂H_t)^T X_t - Input-to-state weight gradients
        backward_pass_ssm(slm->ssms[layer], layer_input);
        
        // Compute gradients for the input to this layer (except for first layer)
        // This implements: ∂L/∂X_{layer} = ∂L/∂H_{layer+1} B_{layer+1}^T + ∂L/∂Y_{layer+1} D_{layer+1}^T
        if (layer > 0) {
            int total_elements = slm->ssms[layer]->seq_len * slm->ssms[layer]->batch_size;
            
            // Clear gradient buffer for input gradients
            CHECK_CUDA(cudaMemset(slm->d_ssm_gradients[layer - 1], 0, 
                                 total_elements * slm->ssms[layer]->input_dim * sizeof(float)));
            
            // ∂L/∂X = B^T (∂L/∂H) - Gradient from state path (vectorized across all timesteps)
            // Computes: [input_dim, seq_len*batch_size] = [input_dim, state_dim] * [state_dim, seq_len*batch_size]
            CHECK_CUBLAS(cublasSgemm(slm->ssms[layer]->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    slm->ssms[layer]->input_dim, total_elements, slm->ssms[layer]->state_dim,
                                    &alpha, slm->ssms[layer]->d_B, slm->ssms[layer]->input_dim,
                                    slm->ssms[layer]->d_state_error, slm->ssms[layer]->state_dim,
                                    &beta, slm->d_ssm_gradients[layer - 1], slm->ssms[layer]->input_dim));
            
            // ∂L/∂X += D^T (∂L/∂Y) - Gradient from output path (vectorized across all timesteps)
            // Computes: [input_dim, seq_len*batch_size] += [input_dim, output_dim] * [output_dim, seq_len*batch_size]
            CHECK_CUBLAS(cublasSgemm(slm->ssms[layer]->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    slm->ssms[layer]->input_dim, total_elements, slm->ssms[layer]->output_dim,
                                    &alpha, slm->ssms[layer]->d_D, slm->ssms[layer]->input_dim,
                                    slm->ssms[layer]->d_error, slm->ssms[layer]->output_dim,
                                    &alpha, slm->d_ssm_gradients[layer - 1], slm->ssms[layer]->input_dim));
            
            // Copy computed gradients to previous layer's error buffer for next iteration
            CHECK_CUDA(cudaMemcpy(slm->ssms[layer - 1]->d_error, slm->d_ssm_gradients[layer - 1], 
                                 total_elements * slm->ssms[layer]->input_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    
    // Compute input gradients from first SSM back to embeddings (vectorized across all timesteps)
    // ∂L/∂E_t = ∂L/∂H1_t B1^T + ∂L/∂Y1_t D1^T - Chain rule from SSM1 back to character embeddings
    int total_elements = slm->ssms[0]->seq_len * slm->ssms[0]->batch_size;
    
    // Clear embedding gradient buffer
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         total_elements * slm->ssms[0]->input_dim * sizeof(float)));
    
    // ∂L/∂E += B1^T (∂L/∂H1) - Gradient from state path (vectorized across all timesteps)
    // Computes: [embed_dim, seq_len*batch_size] = [embed_dim, state_dim] * [state_dim, seq_len*batch_size]
    CHECK_CUBLAS(cublasSgemm(slm->ssms[0]->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->ssms[0]->input_dim, total_elements, slm->ssms[0]->state_dim,
                            &alpha, slm->ssms[0]->d_B, slm->ssms[0]->input_dim,
                            slm->ssms[0]->d_state_error, slm->ssms[0]->state_dim,
                            &beta, slm->d_input_gradients, slm->ssms[0]->input_dim));
    
    // ∂L/∂E += D1^T (∂L/∂Y1) - Gradient from output path (vectorized across all timesteps)
    // Computes: [embed_dim, seq_len*batch_size] += [embed_dim, output_dim] * [output_dim, seq_len*batch_size]
    CHECK_CUBLAS(cublasSgemm(slm->ssms[0]->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->ssms[0]->input_dim, total_elements, slm->ssms[0]->output_dim,
                            &alpha, slm->ssms[0]->d_D, slm->ssms[0]->input_dim,
                            slm->ssms[0]->d_error, slm->ssms[0]->output_dim,
                            &alpha, slm->d_input_gradients, slm->ssms[0]->input_dim));
    
    // Accumulate embedding gradients: ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t
    // For each character c, sum gradients from all positions where that character appears
    dim3 block(256);
    dim3 grid((slm->ssms[0]->batch_size + 255) / 256, slm->ssms[0]->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        slm->ssms[0]->batch_size, slm->embed_dim
    );
}

// Update weights using AdamW: W = (1-λη)W - η·m̂/√v̂
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update all SSM weights
    for (int i = 0; i < slm->num_layers; i++) {
        update_weights_ssm(slm->ssms[i], learning_rate);
    }
    
    // Update MLP weights
    update_weights_mlp(slm->mlp, learning_rate);
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssms[0]->beta1, slm->ssms[0]->t);
    float beta2_t = powf(slm->ssms[0]->beta2, slm->ssms[0]->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssms[0]->beta1, slm->ssms[0]->beta2, slm->ssms[0]->epsilon,
        learning_rate, slm->ssms[0]->weight_decay, alpha_t,
        embed_size, slm->ssms[0]->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Save SSMs
    char* dot;
    for (int i = 0; i < slm->num_layers; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_ssm%d.bin", i);
        strcat(ssm_file, suffix);
        save_ssm(slm->ssms[i], ssm_file);
    }
    
    // Save MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    save_mlp(slm->mlp, mlp_file);
    
    // Save embeddings and metadata
    char embed_file[256];
    strcpy(embed_file, filename);
    dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    // Allocate temporary host memory for embeddings
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    float* h_embeddings_m = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    float* h_embeddings_v = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    
    // Copy embeddings from device to host
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_m, slm->d_embeddings_m, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_v, slm->d_embeddings_v, 
                         slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    FILE* f = fopen(embed_file, "wb");
    if (!f) {
        printf("Error opening file for writing: %s\n", embed_file);
        free(h_embeddings);
        free(h_embeddings_m);
        free(h_embeddings_v);
        return;
    }
    
    // Save dimensions and metadata
    fwrite(&slm->vocab_size, sizeof(int), 1, f);
    fwrite(&slm->embed_dim, sizeof(int), 1, f);
    fwrite(&slm->num_layers, sizeof(int), 1, f);
    fwrite(&slm->ssms[0]->seq_len, sizeof(int), 1, f);
    fwrite(&slm->ssms[0]->state_dim, sizeof(int), 1, f);
    fwrite(&slm->ssms[0]->batch_size, sizeof(int), 1, f);
    
    // Save embeddings and Adam state
    fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, f);
    fwrite(h_embeddings_m, sizeof(float), slm->vocab_size * slm->embed_dim, f);
    fwrite(h_embeddings_v, sizeof(float), slm->vocab_size * slm->embed_dim, f);
    
    // Free temporary host memory
    free(h_embeddings);
    free(h_embeddings_m);
    free(h_embeddings_v);
    
    fclose(f);
    printf("Embeddings saved to %s\n", embed_file);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load embeddings and metadata first
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    FILE* f = fopen(embed_file, "rb");
    if (!f) {
        printf("Error opening file for reading: %s\n", embed_file);
        return NULL;
    }
    
    // Read dimensions and metadata
    int vocab_size, embed_dim, num_layers, seq_len, state_dim, stored_batch_size;
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&embed_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&seq_len, sizeof(int), 1, f);
    fread(&state_dim, sizeof(int), 1, f);
    fread(&stored_batch_size, sizeof(int), 1, f);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Allocate temporary host memory for embeddings
    float* h_embeddings = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_embeddings_m = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_embeddings_v = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    
    // Read embeddings and Adam state
    fread(h_embeddings, sizeof(float), vocab_size * embed_dim, f);
    fread(h_embeddings_m, sizeof(float), vocab_size * embed_dim, f);
    fread(h_embeddings_v, sizeof(float), vocab_size * embed_dim, f);
    fclose(f);
    
    // Initialize SLM with loaded parameters
    SLM* slm = init_slm(embed_dim, state_dim, seq_len, num_layers, batch_size);
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                         vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_m, h_embeddings_m, 
                         vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_v, h_embeddings_v, 
                         vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_embeddings);
    free(h_embeddings_m);
    free(h_embeddings_v);
    
    // Load SSMs directly into the allocated structures
    for (int i = 0; i < num_layers; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_ssm%d.bin", i);
        strcat(ssm_file, suffix);
        
        FILE* ssm_f = fopen(ssm_file, "rb");
        if (!ssm_f) {
            printf("Error opening SSM file: %s\n", ssm_file);
            free_slm(slm);
            return NULL;
        }
        
        // Read and verify dimensions
        int input_dim, state_dim_file, output_dim, seq_len_file, stored_batch_size_file;
        fread(&input_dim, sizeof(int), 1, ssm_f);
        fread(&state_dim_file, sizeof(int), 1, ssm_f);
        fread(&output_dim, sizeof(int), 1, ssm_f);
        fread(&seq_len_file, sizeof(int), 1, ssm_f);
        fread(&stored_batch_size_file, sizeof(int), 1, ssm_f);
        
        // Allocate temporary host memory for SSM matrices
        float* A = (float*)malloc(state_dim * state_dim * sizeof(float));
        float* B = (float*)malloc(state_dim * input_dim * sizeof(float));
        float* C = (float*)malloc(output_dim * state_dim * sizeof(float));
        float* D = (float*)malloc(output_dim * input_dim * sizeof(float));
        
        // Read matrices
        fread(A, sizeof(float), state_dim * state_dim, ssm_f);
        fread(B, sizeof(float), state_dim * input_dim, ssm_f);
        fread(C, sizeof(float), output_dim * state_dim, ssm_f);
        fread(D, sizeof(float), output_dim * input_dim, ssm_f);
        
        fread(&slm->ssms[i]->t, sizeof(int), 1, ssm_f);
        
        // Copy matrices to device
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_A, A, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_B, B, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_C, C, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_D, D, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(A); free(B); free(C); free(D);
        
        // Load Adam state
        float* A_m = (float*)malloc(state_dim * state_dim * sizeof(float));
        float* A_v = (float*)malloc(state_dim * state_dim * sizeof(float));
        float* B_m = (float*)malloc(state_dim * input_dim * sizeof(float));
        float* B_v = (float*)malloc(state_dim * input_dim * sizeof(float));
        float* C_m = (float*)malloc(output_dim * state_dim * sizeof(float));
        float* C_v = (float*)malloc(output_dim * state_dim * sizeof(float));
        float* D_m = (float*)malloc(output_dim * input_dim * sizeof(float));
        float* D_v = (float*)malloc(output_dim * input_dim * sizeof(float));
        
        fread(A_m, sizeof(float), state_dim * state_dim, ssm_f);
        fread(A_v, sizeof(float), state_dim * state_dim, ssm_f);
        fread(B_m, sizeof(float), state_dim * input_dim, ssm_f);
        fread(B_v, sizeof(float), state_dim * input_dim, ssm_f);
        fread(C_m, sizeof(float), output_dim * state_dim, ssm_f);
        fread(C_v, sizeof(float), output_dim * state_dim, ssm_f);
        fread(D_m, sizeof(float), output_dim * input_dim, ssm_f);
        fread(D_v, sizeof(float), output_dim * input_dim, ssm_f);
        
        // Copy Adam state to device
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_A_m, A_m, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_A_v, A_v, state_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_B_m, B_m, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_B_v, B_v, state_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_C_m, C_m, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_C_v, C_v, output_dim * state_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_D_m, D_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->ssms[i]->d_D_v, D_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(A_m); free(A_v); free(B_m); free(B_v);
        free(C_m); free(C_v); free(D_m); free(D_v);
        
        fclose(ssm_f);
    }
    
    // Load MLP directly
    char mlp_file[256];
    strcpy(mlp_file, filename);
    dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    
    FILE* mlp_f = fopen(mlp_file, "rb");
    if (mlp_f) {
        // Read and verify MLP dimensions
        int input_dim, hidden_dim, output_dim, stored_batch_size_mlp;
        fread(&input_dim, sizeof(int), 1, mlp_f);
        fread(&hidden_dim, sizeof(int), 1, mlp_f);
        fread(&output_dim, sizeof(int), 1, mlp_f);
        fread(&stored_batch_size_mlp, sizeof(int), 1, mlp_f);
        
        // Allocate temporary host memory for MLP weights
        float* W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
        float* W2 = (float*)malloc(output_dim * hidden_dim * sizeof(float));
        
        fread(W1, sizeof(float), hidden_dim * input_dim, mlp_f);
        fread(W2, sizeof(float), output_dim * hidden_dim, mlp_f);
        fread(&slm->mlp->t, sizeof(int), 1, mlp_f);
        
        // Copy MLP weights to device
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W1, W1, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W2, W2, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(W1); free(W2);
        
        // Load MLP Adam state
        float* W1_m = (float*)malloc(hidden_dim * input_dim * sizeof(float));
        float* W1_v = (float*)malloc(hidden_dim * input_dim * sizeof(float));
        float* W2_m = (float*)malloc(output_dim * hidden_dim * sizeof(float));
        float* W2_v = (float*)malloc(output_dim * hidden_dim * sizeof(float));
        
        fread(W1_m, sizeof(float), hidden_dim * input_dim, mlp_f);
        fread(W1_v, sizeof(float), hidden_dim * input_dim, mlp_f);
        fread(W2_m, sizeof(float), output_dim * hidden_dim, mlp_f);
        fread(W2_v, sizeof(float), output_dim * hidden_dim, mlp_f);
        
        // Copy MLP Adam state to device
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W1_m, W1_m, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W1_v, W1_v, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W2_m, W2_m, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(slm->mlp->d_W2_v, W2_v, output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
        
        free(W1_m); free(W1_v); free(W2_m); free(W2_v);
        fclose(mlp_f);
    }
    
    printf("Model loaded from %s (with %d SSM layers)\n", filename, num_layers);
    return slm;
}

// Text generation function
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature, float top_p) {
    int seed_len = strlen(seed_text);
    if (seed_len == 0) {
        printf("Error: Empty seed text\n");
        return;
    }
    
    // Create a temporary SLM instance for generation with batch_size=1
    int max_timesteps = seed_len + generation_length;
    SLM* gen_slm = init_slm(slm->embed_dim, slm->ssms[0]->state_dim, max_timesteps, slm->num_layers, 1);
    
    // Copy trained weights from main model to generation model
    // Copy all SSM weights
    for (int i = 0; i < slm->num_layers; i++) {
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_A, slm->ssms[i]->d_A, 
                             slm->ssms[i]->state_dim * slm->ssms[i]->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_B, slm->ssms[i]->d_B, 
                             slm->ssms[i]->state_dim * slm->ssms[i]->input_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_C, slm->ssms[i]->d_C, 
                             slm->ssms[i]->output_dim * slm->ssms[i]->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_D, slm->ssms[i]->d_D, 
                             slm->ssms[i]->output_dim * slm->ssms[i]->input_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    } 
    
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
    for (int i = 0; i < gen_slm->num_layers; i++) {
        reset_state_ssm(gen_slm->ssms[i]);
    }
    
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
        
        // Forward through all SSM layers using loop
        float* current_input = gen_slm->d_embedded_input;
        for (int layer = 0; layer < gen_slm->num_layers; layer++) {
            // H_t = X_t B^T + H_{t-1} A^T
            // O_t = H_t σ(H_t)
            // Y_t = O_t C^T + X_t D^T
            forward_pass_ssm(gen_slm->ssms[layer], current_input, i);
            
            // For all but the last layer, copy output to intermediate buffer for next layer
            if (layer < gen_slm->num_layers - 1) {
                float* d_output_t = gen_slm->ssms[layer]->d_predictions + i * 1 * gen_slm->ssms[layer]->output_dim;
                float* d_ssm_out_t = gen_slm->d_ssm_outputs[layer] + i * 1 * gen_slm->embed_dim;
                CHECK_CUDA(cudaMemcpy(d_ssm_out_t, d_output_t, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = d_ssm_out_t;
            }
        }
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
        
        // Forward through all SSM layers using loop
        int timestep = seed_len + i;
        float* current_input = gen_slm->d_embedded_input;
        for (int layer = 0; layer < gen_slm->num_layers; layer++) {
            // H_t = X_t B^T + H_{t-1} A^T
            // O_t = H_t σ(H_t)
            // Y_t = O_t C^T + X_t D^T
            forward_pass_ssm(gen_slm->ssms[layer], current_input, timestep);
            
            // For all but the last layer, copy output to intermediate buffer for next layer
            if (layer < gen_slm->num_layers - 1) {
                float* d_output_t = gen_slm->ssms[layer]->d_predictions + timestep * 1 * gen_slm->ssms[layer]->output_dim;
                float* d_ssm_out_t = gen_slm->d_ssm_outputs[layer] + timestep * 1 * gen_slm->embed_dim;
                CHECK_CUDA(cudaMemcpy(d_ssm_out_t, d_output_t, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = d_ssm_out_t;
            }
        }
        
        // Get the last SSM output for this timestep
        float* d_y_last_t = gen_slm->ssms[gen_slm->num_layers - 1]->d_predictions + timestep * 1 * gen_slm->ssms[gen_slm->num_layers - 1]->output_dim;
        
        // Z_t = Y_last_t W_1
        // A_t = Z_t σ(Z_t)
        // L_t = A_t W_2 - Forward through MLP
        forward_pass_mlp(gen_slm->mlp, d_y_last_t);
        
        // P_t = softmax(L_t) - Apply softmax to get probabilities
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
        
        // Apply nucleus sampling if top_p < 1.0
        if (top_p < 1.0f && top_p > 0.0f) {
            // Create array of indices for sorting
            int* indices = (int*)malloc(gen_slm->vocab_size * sizeof(int));
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                indices[j] = j;
            }
            
            // Sort indices by probability (descending order)
            for (int i = 0; i < gen_slm->vocab_size - 1; i++) {
                for (int j = i + 1; j < gen_slm->vocab_size; j++) {
                    if (h_probs[indices[i]] < h_probs[indices[j]]) {
                        int temp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = temp;
                    }
                }
            }
            
            // Find the cutoff point where cumulative probability >= top_p
            float cumulative_prob = 0.0f;
            int cutoff = 0;
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                cumulative_prob += h_probs[indices[j]];
                if (cumulative_prob >= top_p) {
                    cutoff = j + 1;
                    break;
                }
            }
            
            // Zero out probabilities for tokens not in the nucleus set
            for (int j = cutoff; j < gen_slm->vocab_size; j++) {
                h_probs[indices[j]] = 0.0f;
            }
            
            // Renormalize the remaining probabilities
            float sum = 0.0f;
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                sum += h_probs[j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    h_probs[j] /= sum;
                }
            }
            
            free(indices);
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