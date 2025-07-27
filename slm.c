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

    // Initialize 4 SSM layers (all embed_dim -> embed_dim)
    for (int i = 0; i < 4; i++) {
        slm->ssm[i] = init_ssm(embed_dim, state_dim, embed_dim, seq_len, batch_size);
    }
    
    // Initialize 4 MLP layers (all embed_dim -> embed_dim, except last one outputs vocab_size)
    for (int i = 0; i < 3; i++) {
        slm->mlp[i] = init_mlp(embed_dim, 4 * embed_dim, embed_dim, seq_len * batch_size);
    }
    // Last MLP layer outputs vocabulary size
    slm->mlp[3] = init_mlp(embed_dim, 4 * embed_dim, slm->vocab_size, seq_len * batch_size);

    // Allocate embedding matrices
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Allocate working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * embed_dim * sizeof(float)));
    
    // Allocate intermediate layer outputs
    for (int i = 0; i < 7; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[i], seq_len * batch_size * embed_dim * sizeof(float)));
    }
    // Last layer outputs vocab_size
    CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[7], seq_len * batch_size * slm->vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, seq_len * batch_size * embed_dim * sizeof(float)));
    
    // Allocate gradient buffers for each layer
    for (int i = 0; i < 7; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_layer_gradients[i], seq_len * batch_size * embed_dim * sizeof(float)));
    }
    // Last layer has vocab_size gradients
    CHECK_CUDA(cudaMalloc(&slm->d_layer_gradients[7], seq_len * batch_size * slm->vocab_size * sizeof(float)));
    
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
        // Free all SSM layers
        for (int i = 0; i < 4; i++) {
            if (slm->ssm[i]) free_ssm(slm->ssm[i]);
        }
        
        // Free all MLP layers
        for (int i = 0; i < 4; i++) {
            if (slm->mlp[i]) free_mlp(slm->mlp[i]);
        }
        
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        
        // Free layer outputs
        for (int i = 0; i < 8; i++) {
            cudaFree(slm->d_layer_outputs[i]);
        }
        
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        
        // Free layer gradients
        for (int i = 0; i < 8; i++) {
            cudaFree(slm->d_layer_gradients[i]);
        }
        
        cudaFree(slm->d_losses);
        free(slm);
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm[0]->seq_len;
    int batch_size = slm->ssm[0]->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // Current input starts with embeddings
    float* current_input = slm->d_embedded_input;
    
    // Process through alternating SSM and MLP layers: SSM->MLP->SSM->MLP->SSM->MLP->SSM->MLP
    for (int layer = 0; layer < 4; layer++) {
        // SSM layer
        reset_state_ssm(slm->ssm[layer]);
        for (int t = 0; t < seq_len; t++) {
            float* d_input_t = current_input + t * batch_size * slm->embed_dim;
            forward_pass_ssm(slm->ssm[layer], d_input_t, t);
        }
        
        // Copy SSM output to layer output buffer
        int ssm_output_size = seq_len * batch_size * slm->embed_dim;
        CHECK_CUDA(cudaMemcpy(slm->d_layer_outputs[layer * 2], slm->ssm[layer]->d_predictions, 
                             ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // MLP layer
        if (layer < 3) {
            // First 3 MLP layers output embed_dim
            forward_pass_mlp(slm->mlp[layer], slm->d_layer_outputs[layer * 2]);
            
            // Copy MLP output to next layer input buffer
            CHECK_CUDA(cudaMemcpy(slm->d_layer_outputs[layer * 2 + 1], slm->mlp[layer]->d_predictions, 
                                 ssm_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // Update current input for next SSM layer
            current_input = slm->d_layer_outputs[layer * 2 + 1];
        } else {
            // Last MLP layer outputs vocab_size
            forward_pass_mlp(slm->mlp[layer], slm->d_layer_outputs[layer * 2]);
            
            // Copy MLP output to final output buffer (vocab_size)
            int final_output_size = seq_len * batch_size * slm->vocab_size;
            CHECK_CUDA(cudaMemcpy(slm->d_layer_outputs[layer * 2 + 1], slm->mlp[layer]->d_predictions, 
                                 final_output_size * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    
    // P_t = softmax(L_t) - Apply softmax for probability distribution
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->d_layer_outputs[7], total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm[0]->seq_len;
    int batch_size = slm->ssm[0]->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // ∂L/∂L_t = P_t - 1_{y_t} - Compute cross-entropy gradient (softmax - one_hot) for backprop
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp[3]->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // L = -log(P_{y_t}) - Calculate actual cross-entropy loss
    cross_entropy_loss_kernel<<<blocks, 256>>>(
        slm->d_losses, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Sum all losses
    float total_loss;
    CHECK_CUBLAS(cublasSasum(slm->ssm[0]->cublas_handle, total_tokens, 
                            slm->d_losses, 1, &total_loss));
    
    return total_loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    // Zero all SSM gradients
    for (int i = 0; i < 4; i++) {
        zero_gradients_ssm(slm->ssm[i]);
    }
    
    // Zero all MLP gradients
    for (int i = 0; i < 4; i++) {
        zero_gradients_mlp(slm->mlp[i]);
    }
    
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm[0]->seq_len;
    int batch_size = slm->ssm[0]->batch_size;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Backward pass through layers in reverse order: MLP[3] -> SSM[3] -> MLP[2] -> SSM[2] -> MLP[1] -> SSM[1] -> MLP[0] -> SSM[0]
    
    // Layer 7: MLP[3] backward (final layer, gradient already set in calculate_loss)
    backward_pass_mlp(slm->mlp[3], slm->d_layer_outputs[6]); // Input was SSM[3] output
    
    // Copy MLP[3] input gradients to SSM[3] error
    int embed_elements = seq_len * batch_size * slm->embed_dim;
    CHECK_CUDA(cudaMemcpy(slm->ssm[3]->d_error, slm->mlp[3]->d_error, 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Layer 6: SSM[3] backward
    backward_pass_ssm(slm->ssm[3], slm->d_layer_outputs[5]); // Input was MLP[2] output
    
    // Compute gradients with respect to SSM[3] input (MLP[2] output)
    CHECK_CUDA(cudaMemset(slm->d_layer_gradients[5], 0, embed_elements * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_layer_grad_t = slm->d_layer_gradients[5] + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm[3]->d_state_error + t * batch_size * slm->ssm[3]->state_dim;
        float* d_output_error_t = slm->ssm[3]->d_error + t * batch_size * slm->ssm[3]->output_dim;
        
        // Gradient from state path: B^T * state_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[3]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[3]->input_dim, batch_size, slm->ssm[3]->state_dim,
                                &alpha, slm->ssm[3]->d_B, slm->ssm[3]->input_dim,
                                d_state_error_t, slm->ssm[3]->state_dim,
                                &beta, d_layer_grad_t, slm->ssm[3]->input_dim));
        
        // Gradient from output path: D^T * output_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[3]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[3]->input_dim, batch_size, slm->ssm[3]->output_dim,
                                &alpha, slm->ssm[3]->d_D, slm->ssm[3]->input_dim,
                                d_output_error_t, slm->ssm[3]->output_dim,
                                &alpha, d_layer_grad_t, slm->ssm[3]->input_dim));
    }
    
    // Layer 5: MLP[2] backward
    CHECK_CUDA(cudaMemcpy(slm->mlp[2]->d_error, slm->d_layer_gradients[5], 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_mlp(slm->mlp[2], slm->d_layer_outputs[4]); // Input was SSM[2] output
    
    // Layer 4: SSM[2] backward
    CHECK_CUDA(cudaMemcpy(slm->ssm[2]->d_error, slm->mlp[2]->d_error, 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_ssm(slm->ssm[2], slm->d_layer_outputs[3]); // Input was MLP[1] output
    
    // Compute gradients with respect to SSM[2] input (MLP[1] output)
    CHECK_CUDA(cudaMemset(slm->d_layer_gradients[3], 0, embed_elements * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_layer_grad_t = slm->d_layer_gradients[3] + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm[2]->d_state_error + t * batch_size * slm->ssm[2]->state_dim;
        float* d_output_error_t = slm->ssm[2]->d_error + t * batch_size * slm->ssm[2]->output_dim;
        
        // Gradient from state path: B^T * state_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[2]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[2]->input_dim, batch_size, slm->ssm[2]->state_dim,
                                &alpha, slm->ssm[2]->d_B, slm->ssm[2]->input_dim,
                                d_state_error_t, slm->ssm[2]->state_dim,
                                &beta, d_layer_grad_t, slm->ssm[2]->input_dim));
        
        // Gradient from output path: D^T * output_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[2]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[2]->input_dim, batch_size, slm->ssm[2]->output_dim,
                                &alpha, slm->ssm[2]->d_D, slm->ssm[2]->input_dim,
                                d_output_error_t, slm->ssm[2]->output_dim,
                                &alpha, d_layer_grad_t, slm->ssm[2]->input_dim));
    }
    
    // Layer 3: MLP[1] backward
    CHECK_CUDA(cudaMemcpy(slm->mlp[1]->d_error, slm->d_layer_gradients[3], 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_mlp(slm->mlp[1], slm->d_layer_outputs[2]); // Input was SSM[1] output
    
    // Layer 2: SSM[1] backward
    CHECK_CUDA(cudaMemcpy(slm->ssm[1]->d_error, slm->mlp[1]->d_error, 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_ssm(slm->ssm[1], slm->d_layer_outputs[1]); // Input was MLP[0] output
    
    // Compute gradients with respect to SSM[1] input (MLP[0] output)
    CHECK_CUDA(cudaMemset(slm->d_layer_gradients[1], 0, embed_elements * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_layer_grad_t = slm->d_layer_gradients[1] + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm[1]->d_state_error + t * batch_size * slm->ssm[1]->state_dim;
        float* d_output_error_t = slm->ssm[1]->d_error + t * batch_size * slm->ssm[1]->output_dim;
        
        // Gradient from state path: B^T * state_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[1]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[1]->input_dim, batch_size, slm->ssm[1]->state_dim,
                                &alpha, slm->ssm[1]->d_B, slm->ssm[1]->input_dim,
                                d_state_error_t, slm->ssm[1]->state_dim,
                                &beta, d_layer_grad_t, slm->ssm[1]->input_dim));
        
        // Gradient from output path: D^T * output_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[1]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[1]->input_dim, batch_size, slm->ssm[1]->output_dim,
                                &alpha, slm->ssm[1]->d_D, slm->ssm[1]->input_dim,
                                d_output_error_t, slm->ssm[1]->output_dim,
                                &alpha, d_layer_grad_t, slm->ssm[1]->input_dim));
    }
    
    // Layer 1: MLP[0] backward
    CHECK_CUDA(cudaMemcpy(slm->mlp[0]->d_error, slm->d_layer_gradients[1], 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_mlp(slm->mlp[0], slm->d_layer_outputs[0]); // Input was SSM[0] output
    
    // Layer 0: SSM[0] backward
    CHECK_CUDA(cudaMemcpy(slm->ssm[0]->d_error, slm->mlp[0]->d_error, 
                         embed_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    backward_pass_ssm(slm->ssm[0], slm->d_embedded_input); // Input was embeddings
    
    // Compute gradients with respect to embeddings
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, embed_elements * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm[0]->d_state_error + t * batch_size * slm->ssm[0]->state_dim;
        float* d_output_error_t = slm->ssm[0]->d_error + t * batch_size * slm->ssm[0]->output_dim;
        
        // Gradient from state path: B^T * state_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[0]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[0]->input_dim, batch_size, slm->ssm[0]->state_dim,
                                &alpha, slm->ssm[0]->d_B, slm->ssm[0]->input_dim,
                                d_state_error_t, slm->ssm[0]->state_dim,
                                &beta, d_input_grad_t, slm->ssm[0]->input_dim));
        
        // Gradient from output path: D^T * output_error
        CHECK_CUBLAS(cublasSgemm(slm->ssm[0]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm[0]->input_dim, batch_size, slm->ssm[0]->output_dim,
                                &alpha, slm->ssm[0]->d_D, slm->ssm[0]->input_dim,
                                d_output_error_t, slm->ssm[0]->output_dim,
                                &alpha, d_input_grad_t, slm->ssm[0]->input_dim));
    }
    
    // Accumulate embedding gradients
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
    for (int i = 0; i < 4; i++) {
        update_weights_ssm(slm->ssm[i], learning_rate);
    }
    
    // Update all MLP weights
    for (int i = 0; i < 4; i++) {
        update_weights_mlp(slm->mlp[i], learning_rate);
    }
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssm[0]->beta1, slm->ssm[0]->t);
    float beta2_t = powf(slm->ssm[0]->beta2, slm->ssm[0]->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssm[0]->beta1, slm->ssm[0]->beta2, slm->ssm[0]->epsilon,
        learning_rate, slm->ssm[0]->weight_decay, alpha_t,
        embed_size, slm->ssm[0]->batch_size
    );
}

// Save model
void save_slm(SLM* slm, const char* filename) {
    // Save all SSM layers
    for (int i = 0; i < 4; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        char* dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_ssm%d.bin", i);
        strcat(ssm_file, suffix);
        save_ssm(slm->ssm[i], ssm_file);
    }
    
    // Save all MLP layers
    for (int i = 0; i < 4; i++) {
        char mlp_file[256];
        strcpy(mlp_file, filename);
        char* dot = strrchr(mlp_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_mlp%d.bin", i);
        strcat(mlp_file, suffix);
        save_mlp(slm->mlp[i], mlp_file);
    }
    
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
    // Load all SSM layers
    SSM* ssms[4];
    for (int i = 0; i < 4; i++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        char* dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_ssm%d.bin", i);
        strcat(ssm_file, suffix);
        ssms[i] = load_ssm(ssm_file, custom_batch_size);
        if (!ssms[i]) {
            // Clean up already loaded SSMs
            for (int j = 0; j < i; j++) {
                free_ssm(ssms[j]);
            }
            return NULL;
        }
    }
    
    // Load all MLP layers
    MLP* mlps[4];
    for (int i = 0; i < 4; i++) {
        char mlp_file[256];
        strcpy(mlp_file, filename);
        char* dot = strrchr(mlp_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_mlp%d.bin", i);
        strcat(mlp_file, suffix);
        mlps[i] = load_mlp(mlp_file, ssms[0]->seq_len * ssms[0]->batch_size);
        if (!mlps[i]) {
            // Clean up already loaded components
            for (int j = 0; j < 4; j++) {
                free_ssm(ssms[j]);
            }
            for (int j = 0; j < i; j++) {
                free_mlp(mlps[j]);
            }
            return NULL;
        }
    }
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Copy loaded components
    for (int i = 0; i < 4; i++) {
        slm->ssm[i] = ssms[i];
        slm->mlp[i] = mlps[i];
    }
    
    slm->vocab_size = 256; // Assuming vocab_size is always 256
    slm->embed_dim = ssms[0]->input_dim;
    
    int seq_len = ssms[0]->seq_len;
    int batch_size = ssms[0]->batch_size;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * slm->embed_dim * sizeof(float)));
    
    // Allocate layer outputs
    for (int i = 0; i < 7; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[i], seq_len * batch_size * slm->embed_dim * sizeof(float)));
    }
    CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[7], seq_len * batch_size * slm->vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, seq_len * batch_size * slm->embed_dim * sizeof(float)));
    
    // Allocate layer gradients
    for (int i = 0; i < 7; i++) {
        CHECK_CUDA(cudaMalloc(&slm->d_layer_gradients[i], seq_len * batch_size * slm->embed_dim * sizeof(float)));
    }
    CHECK_CUDA(cudaMalloc(&slm->d_layer_gradients[7], seq_len * batch_size * slm->vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&slm->d_losses, seq_len * batch_size * sizeof(float)));
    
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
    
    // Create a temporary SLM instance for generation with batch_size=1
    int max_timesteps = seed_len + generation_length;
    SLM* gen_slm = init_slm(slm->embed_dim, slm->ssm[0]->state_dim, max_timesteps, 1);
    
    // Copy trained weights from main model to generation model
    // Copy all SSM weights
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA(cudaMemcpy(gen_slm->ssm[i]->d_A, slm->ssm[i]->d_A, 
                             slm->ssm[i]->state_dim * slm->ssm[i]->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssm[i]->d_B, slm->ssm[i]->d_B, 
                             slm->ssm[i]->state_dim * slm->ssm[i]->input_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssm[i]->d_C, slm->ssm[i]->d_C, 
                             slm->ssm[i]->output_dim * slm->ssm[i]->state_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssm[i]->d_D, slm->ssm[i]->d_D, 
                             slm->ssm[i]->output_dim * slm->ssm[i]->input_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
    }
    
    // Copy all MLP weights
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA(cudaMemcpy(gen_slm->mlp[i]->d_W1, slm->mlp[i]->d_W1,
                             slm->mlp[i]->hidden_dim * slm->mlp[i]->input_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->mlp[i]->d_W2, slm->mlp[i]->d_W2,
                             slm->mlp[i]->output_dim * slm->mlp[i]->hidden_dim * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
    
    // Copy embeddings
    CHECK_CUDA(cudaMemcpy(gen_slm->d_embeddings, slm->d_embeddings,
                         slm->vocab_size * slm->embed_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    // Allocate temporary buffers for generation
    unsigned char* h_input = (unsigned char*)malloc(sizeof(unsigned char));
    unsigned char* d_input;
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    
    // Reset all SSM states for generation
    for (int i = 0; i < 4; i++) {
        reset_state_ssm(gen_slm->ssm[i]);
    }
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text to build up hidden states
    for (int i = 0; i < seed_len; i++) {
        h_input[0] = (unsigned char)seed_text[i];
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // E_t = W_E[X_t] - Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            gen_slm->d_embedded_input, gen_slm->d_embeddings, d_input, 1, gen_slm->embed_dim
        );
        
        // Process through all alternating SSM and MLP layers
        float* current_input = gen_slm->d_embedded_input;
        
        for (int layer = 0; layer < 4; layer++) {
            // SSM layer
            forward_pass_ssm(gen_slm->ssm[layer], current_input, i);
            
            // Get SSM output for this timestep
            float* d_ssm_output_t = gen_slm->ssm[layer]->d_predictions + i * 1 * gen_slm->ssm[layer]->output_dim;
            CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2], d_ssm_output_t, 
                                 gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // MLP layer
            if (layer < 3) {
                // First 3 MLP layers process embed_dim -> embed_dim
                forward_pass_mlp(gen_slm->mlp[layer], gen_slm->d_layer_outputs[layer * 2]);
                CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2 + 1], gen_slm->mlp[layer]->d_predictions, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = gen_slm->d_layer_outputs[layer * 2 + 1];
            } else {
                // Last MLP layer processes embed_dim -> vocab_size
                forward_pass_mlp(gen_slm->mlp[layer], gen_slm->d_layer_outputs[layer * 2]);
                CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2 + 1], gen_slm->mlp[layer]->d_predictions, 
                                     gen_slm->vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
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
        
        // Process through all alternating SSM and MLP layers
        float* current_input = gen_slm->d_embedded_input;
        int timestep = seed_len + i;
        
        for (int layer = 0; layer < 4; layer++) {
            // SSM layer
            forward_pass_ssm(gen_slm->ssm[layer], current_input, timestep);
            
            // Get SSM output for this timestep
            float* d_ssm_output_t = gen_slm->ssm[layer]->d_predictions + timestep * 1 * gen_slm->ssm[layer]->output_dim;
            CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2], d_ssm_output_t, 
                                 gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            
            // MLP layer  
            if (layer < 3) {
                // First 3 MLP layers process embed_dim -> embed_dim
                forward_pass_mlp(gen_slm->mlp[layer], gen_slm->d_layer_outputs[layer * 2]);
                CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2 + 1], gen_slm->mlp[layer]->d_predictions, 
                                     gen_slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
                current_input = gen_slm->d_layer_outputs[layer * 2 + 1];
            } else {
                // Last MLP layer processes embed_dim -> vocab_size
                forward_pass_mlp(gen_slm->mlp[layer], gen_slm->d_layer_outputs[layer * 2]);
                CHECK_CUDA(cudaMemcpy(gen_slm->d_layer_outputs[layer * 2 + 1], gen_slm->mlp[layer]->d_predictions, 
                                     gen_slm->vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 256>>>(
            gen_slm->d_softmax, gen_slm->d_layer_outputs[7], 1, gen_slm->vocab_size
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