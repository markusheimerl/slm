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
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int batch_size, int num_ssm_layers) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
        
    // Set dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    slm->num_ssm_layers = num_ssm_layers;

    // Allocate array of SSM layer pointers
    slm->ssm_layers = (SSM**)malloc(num_ssm_layers * sizeof(SSM*));
    
    // Initialize SSM layers (embed_dim -> embed_dim for all layers)
    for (int i = 0; i < num_ssm_layers; i++) {
        slm->ssm_layers[i] = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
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
    
    // Allocate arrays for intermediate SSM outputs and gradients (excluding the last layer)
    slm->d_ssm_outputs = (float**)malloc((num_ssm_layers - 1) * sizeof(float*));
    slm->d_ssm_gradients = (float**)malloc((num_ssm_layers - 1) * sizeof(float*));
    for (int i = 0; i < num_ssm_layers - 1; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_ssm_outputs[i], seq_len * batch_size * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slm->d_ssm_gradients[i], seq_len * batch_size * embed_dim * sizeof(float)));
    }
    
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
        // Free SSM layers
        if (slm->ssm_layers) {
            for (int i = 0; i < slm->num_ssm_layers; i++) {
                if (slm->ssm_layers[i]) free_ssm(slm->ssm_layers[i]);
            }
            free(slm->ssm_layers);
        }
        
        if (slm->mlp) free_mlp(slm->mlp);
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        
        // Free intermediate buffers
        if (slm->d_ssm_outputs) {
            for (int i = 0; i < slm->num_ssm_layers - 1; i++) {
                cudaFree(slm->d_ssm_outputs[i]);
            }
            free(slm->d_ssm_outputs);
        }
        if (slm->d_ssm_gradients) {
            for (int i = 0; i < slm->num_ssm_layers - 1; i++) {
                cudaFree(slm->d_ssm_gradients[i]);
            }
            free(slm->d_ssm_gradients);
        }
        
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_losses);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm_layers[0]->seq_len;
    int batch_size = slm->ssm_layers[0]->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // Forward through all SSM layers sequentially
    float* current_input = slm->d_embedded_input;
    
    for (int layer = 0; layer < slm->num_ssm_layers; layer++) {
        SSM* current_ssm = slm->ssm_layers[layer];
        
        // Reset state for this layer
        reset_state_ssm(current_ssm);
        
        // Forward pass through current SSM layer for all timesteps
        for (int t = 0; t < seq_len; t++) {
            float* input_t = current_input + t * batch_size * slm->embed_dim;
            forward_pass_ssm(current_ssm, input_t, t);
        }
        
        // If not the last layer, copy output to intermediate buffer
        if (layer < slm->num_ssm_layers - 1) {
            int output_size = seq_len * batch_size * slm->embed_dim;
            CHECK_CUDA(cudaMemcpy(slm->d_ssm_outputs[layer], current_ssm->d_predictions, 
                                 output_size * sizeof(float), cudaMemcpyDeviceToDevice));
            current_input = slm->d_ssm_outputs[layer];
        }
    }

    // Forward through MLP using the last SSM layer's output
    forward_pass_mlp(slm->mlp, slm->ssm_layers[slm->num_ssm_layers - 1]->d_predictions);
    
    // Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm_layers[0]->seq_len;
    int batch_size = slm->ssm_layers[0]->batch_size;
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
    CHECK_CUBLAS(cublasSasum(slm->ssm_layers[0]->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &total_loss));
    
    return total_loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    // Zero gradients for all SSM layers
    for (int i = 0; i < slm->num_ssm_layers; i++) {
        zero_gradients_ssm(slm->ssm_layers[i]);
    }
    zero_gradients_mlp(slm->mlp);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    // Backward through MLP
    SSM* last_ssm = slm->ssm_layers[slm->num_ssm_layers - 1];
    backward_pass_mlp(slm->mlp, last_ssm->d_predictions);
    
    // Copy MLP input gradients to last SSM output error
    int total_elements = last_ssm->seq_len * last_ssm->batch_size * last_ssm->output_dim;
    CHECK_CUDA(cudaMemcpy(last_ssm->d_error, slm->mlp->d_error, 
                         total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backward through SSM layers in reverse order
    for (int layer = slm->num_ssm_layers - 1; layer >= 0; layer--) {
        SSM* current_ssm = slm->ssm_layers[layer];
        
        // Determine input data for this layer
        float* layer_input;
        if (layer == 0) {
            // First layer uses embedded input
            layer_input = slm->d_embedded_input;
        } else {
            // Other layers use previous layer output
            layer_input = slm->d_ssm_outputs[layer - 1];
        }
        
        // Backward pass through current SSM layer
        backward_pass_ssm(current_ssm, layer_input);
        
        // If not the first layer, compute gradients for the previous layer's output
        if (layer > 0) {
            SSM* prev_ssm = slm->ssm_layers[layer - 1];
            float* grad_buffer = slm->d_ssm_gradients[layer - 1];
            
            // Zero the gradient buffer
            CHECK_CUDA(cudaMemset(grad_buffer, 0, 
                                 current_ssm->seq_len * current_ssm->batch_size * current_ssm->input_dim * sizeof(float)));
            
            // Compute gradients for each timestep
            for (int t = 0; t < current_ssm->seq_len; t++) {
                float* grad_t = grad_buffer + t * current_ssm->batch_size * current_ssm->input_dim;
                float* state_error_t = current_ssm->d_state_error + t * current_ssm->batch_size * current_ssm->state_dim;
                float* output_error_t = current_ssm->d_error + t * current_ssm->batch_size * current_ssm->output_dim;
                
                // Gradient from state path: B^T * state_error
                CHECK_CUBLAS(cublasSgemm(current_ssm->cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        current_ssm->input_dim, current_ssm->batch_size, current_ssm->state_dim,
                                        &alpha, current_ssm->d_B, current_ssm->input_dim,
                                        state_error_t, current_ssm->state_dim,
                                        &beta, grad_t, current_ssm->input_dim));
                
                // Gradient from output path: D^T * output_error
                CHECK_CUBLAS(cublasSgemm(current_ssm->cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        current_ssm->input_dim, current_ssm->batch_size, current_ssm->output_dim,
                                        &alpha, current_ssm->d_D, current_ssm->input_dim,
                                        output_error_t, current_ssm->output_dim,
                                        &alpha, grad_t, current_ssm->input_dim));
            }
            
            // Copy gradients to previous SSM's error buffer
            CHECK_CUDA(cudaMemcpy(prev_ssm->d_error, grad_buffer, 
                                 total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    
    // Compute input gradients from first SSM layer for embedding gradients
    SSM* first_ssm = slm->ssm_layers[0];
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         first_ssm->seq_len * first_ssm->batch_size * first_ssm->input_dim * sizeof(float)));
    
    for (int t = 0; t < first_ssm->seq_len; t++) {
        float* input_grad_t = slm->d_input_gradients + t * first_ssm->batch_size * first_ssm->input_dim;
        float* state_error_t = first_ssm->d_state_error + t * first_ssm->batch_size * first_ssm->state_dim;
        float* output_error_t = first_ssm->d_error + t * first_ssm->batch_size * first_ssm->output_dim;
        
        // Gradient from state path: B^T * state_error
        CHECK_CUBLAS(cublasSgemm(first_ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                first_ssm->input_dim, first_ssm->batch_size, first_ssm->state_dim,
                                &alpha, first_ssm->d_B, first_ssm->input_dim,
                                state_error_t, first_ssm->state_dim,
                                &beta, input_grad_t, first_ssm->input_dim));
        
        // Gradient from output path: D^T * output_error
        CHECK_CUBLAS(cublasSgemm(first_ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                first_ssm->input_dim, first_ssm->batch_size, first_ssm->output_dim,
                                &alpha, first_ssm->d_D, first_ssm->input_dim,
                                output_error_t, first_ssm->output_dim,
                                &alpha, input_grad_t, first_ssm->input_dim));
    }
    
    // Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((first_ssm->batch_size + 255) / 256, first_ssm->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        first_ssm->batch_size, slm->embed_dim
    );
}

// Update weights using AdamW: W = (1-λη)W - η·m̂/√v̂
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update all SSM weights
    for (int i = 0; i < slm->num_ssm_layers; i++) {
        update_weights_ssm(slm->ssm_layers[i], learning_rate);
    }
    
    // Update MLP weights
    update_weights_mlp(slm->mlp, learning_rate);
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    // Use first SSM layer for optimization parameters
    SSM* ref_ssm = slm->ssm_layers[0];
    float beta1_t = powf(ref_ssm->beta1, ref_ssm->t);
    float beta2_t = powf(ref_ssm->beta2, ref_ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        ref_ssm->beta1, ref_ssm->beta2, ref_ssm->epsilon,
        learning_rate, ref_ssm->weight_decay, alpha_t,
        embed_size, ref_ssm->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Save first SSM with the original filename
    save_ssm(slm->ssm_layers[0], filename);
    
    // Save additional SSM layers with numbered suffixes
    for (int i = 1; i < slm->num_ssm_layers; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        char* dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        sprintf(ssm_file + strlen(ssm_file), "_ssm%d.bin", i + 1);
        save_ssm(slm->ssm_layers[i], ssm_file);
    }
    
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
        fwrite(&slm->num_ssm_layers, sizeof(int), 1, f); // Save number of layers
        fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, f);
        fclose(f);
        printf("Embeddings saved to %s\n", embed_file);
    }
    
    free(h_embeddings);
}

// Load model
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load embeddings first to get the number of layers
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    int vocab_size, embed_dim, num_ssm_layers = 3; // Default to 3 for backward compatibility
    FILE* f = fopen(embed_file, "rb");
    if (f) {
        fread(&vocab_size, sizeof(int), 1, f);
        fread(&embed_dim, sizeof(int), 1, f);
        // Try to read the number of layers (may not exist in older files)
        if (fread(&num_ssm_layers, sizeof(int), 1, f) != 1) {
            num_ssm_layers = 3; // Default for backward compatibility
            fseek(f, -(long)sizeof(int), SEEK_CUR); // Rewind if read failed
        }
        fclose(f);
    } else {
        printf("Warning: Could not read embeddings file, using defaults\n");
        return NULL;
    }
    
    // Load first SSM to get dimensions
    SSM* first_ssm = load_ssm(filename, custom_batch_size);
    if (!first_ssm) return NULL;
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->num_ssm_layers = num_ssm_layers;
    slm->vocab_size = vocab_size;
    slm->embed_dim = embed_dim;
    
    // Allocate SSM layers array
    slm->ssm_layers = (SSM**)malloc(num_ssm_layers * sizeof(SSM*));
    slm->ssm_layers[0] = first_ssm;
    
    // Load remaining SSM layers
    for (int i = 1; i < num_ssm_layers; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        sprintf(ssm_file + strlen(ssm_file), "_ssm%d.bin", i + 1);
        slm->ssm_layers[i] = load_ssm(ssm_file, custom_batch_size);
        if (!slm->ssm_layers[i]) {
            // Free already loaded layers
            for (int j = 0; j < i; j++) {
                free_ssm(slm->ssm_layers[j]);
            }
            free(slm->ssm_layers);
            free(slm);
            return NULL;
        }
    }
    
    // Load MLP
    char mlp_file[256];
    strcpy(mlp_file, filename);
    char* dot = strrchr(mlp_file, '.');
    if (dot) *dot = '\0';
    strcat(mlp_file, "_mlp.bin");
    slm->mlp = load_mlp(mlp_file, first_ssm->seq_len * first_ssm->batch_size);
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, first_ssm->seq_len * first_ssm->batch_size * first_ssm->input_dim * sizeof(float)));
    
    // Allocate intermediate buffers
    slm->d_ssm_outputs = (float**)malloc((num_ssm_layers - 1) * sizeof(float*));
    slm->d_ssm_gradients = (float**)malloc((num_ssm_layers - 1) * sizeof(float*));
    for (int i = 0; i < num_ssm_layers - 1; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_ssm_outputs[i], first_ssm->seq_len * first_ssm->batch_size * first_ssm->output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slm->d_ssm_gradients[i], first_ssm->seq_len * first_ssm->batch_size * first_ssm->output_dim * sizeof(float)));
    }
    
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, first_ssm->seq_len * first_ssm->batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, first_ssm->seq_len * first_ssm->batch_size * first_ssm->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, first_ssm->seq_len * first_ssm->batch_size * sizeof(float)));
    
    // Load embeddings
    f = fopen(embed_file, "rb");
    if (f) {
        int file_vocab_size, file_embed_dim, file_num_layers;
        fread(&file_vocab_size, sizeof(int), 1, f);
        fread(&file_embed_dim, sizeof(int), 1, f);
        // Try to read number of layers, but don't fail if not present
        if (fread(&file_num_layers, sizeof(int), 1, f) != 1) {
            // Rewind to read embeddings from the correct position
            fseek(f, 2 * sizeof(int), SEEK_SET);
        }
        
        float* h_embeddings = (float*)malloc(file_vocab_size * file_embed_dim * sizeof(float));
        fread(h_embeddings, sizeof(float), file_vocab_size * file_embed_dim, f);
        
        CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, 
                             file_vocab_size * file_embed_dim * sizeof(float), cudaMemcpyHostToDevice));
        
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
    
    // Create a temporary SLM instance for generation with batch_size=1
    int max_timesteps = seed_len + generation_length;
    SLM* gen_slm = init_slm(slm->embed_dim, slm->ssm_layers[0]->state_dim, max_timesteps, 1, slm->num_ssm_layers);
    
    // Copy trained weights from main model to generation model
    for (int i = 0; i < slm->num_ssm_layers; i++) {
        SSM* src_ssm = slm->ssm_layers[i];
        SSM* dst_ssm = gen_slm->ssm_layers[i];
        
        CHECK_CUDA(cudaMemcpy(dst_ssm->d_A, src_ssm->d_A, 
                             src_ssm->state_dim * src_ssm->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_ssm->d_B, src_ssm->d_B, 
                             src_ssm->state_dim * src_ssm->input_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_ssm->d_C, src_ssm->d_C, 
                             src_ssm->output_dim * src_ssm->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_ssm->d_D, src_ssm->d_D, 
                             src_ssm->output_dim * src_ssm->input_dim * sizeof(float), 
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
    for (int i = 0; i < gen_slm->num_ssm_layers; i++) {
        reset_state_ssm(gen_slm->ssm_layers[i]);
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
        
        // Forward through all SSM layers sequentially
        float* current_input = gen_slm->d_embedded_input;
        
        for (int layer = 0; layer < gen_slm->num_ssm_layers; layer++) {
            forward_pass_ssm(gen_slm->ssm_layers[layer], current_input, i);
            
            // If not the last layer, copy output to intermediate buffer
            if (layer < gen_slm->num_ssm_layers - 1) {
                float* d_output_t = gen_slm->ssm_layers[layer]->d_predictions + i * 1 * gen_slm->embed_dim;
                float* d_buffer_t = gen_slm->d_ssm_outputs[layer] + i * 1 * gen_slm->embed_dim;
                CHECK_CUDA(cudaMemcpy(d_buffer_t, d_output_t, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = gen_slm->d_ssm_outputs[layer] + i * 1 * gen_slm->embed_dim;
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
        
        // Forward through all SSM layers sequentially
        float* current_input = gen_slm->d_embedded_input;
        int timestep = seed_len + i;
        
        for (int layer = 0; layer < gen_slm->num_ssm_layers; layer++) {
            forward_pass_ssm(gen_slm->ssm_layers[layer], current_input, timestep);
            
            // If not the last layer, copy output to intermediate buffer
            if (layer < gen_slm->num_ssm_layers - 1) {
                float* d_output_t = gen_slm->ssm_layers[layer]->d_predictions + timestep * 1 * gen_slm->embed_dim;
                float* d_buffer_t = gen_slm->d_ssm_outputs[layer] + timestep * 1 * gen_slm->embed_dim;
                CHECK_CUDA(cudaMemcpy(d_buffer_t, d_output_t, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = gen_slm->d_ssm_outputs[layer] + timestep * 1 * gen_slm->embed_dim;
            }
        }
        
        // Get the last SSM output for this timestep
        float* d_final_output = gen_slm->ssm_layers[gen_slm->num_ssm_layers - 1]->d_predictions + timestep * 1 * gen_slm->embed_dim;
        
        // Forward through MLP
        forward_pass_mlp(gen_slm->mlp, d_final_output);
        
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