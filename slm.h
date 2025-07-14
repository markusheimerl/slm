#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"
#include "mlp/gpu/mlp.h"

typedef struct {
    SSM** ssm_layers;           // Array of SSM layers
    MLP** mlp_layers;           // Array of MLP layers
    int num_layers;             // Number of layers
    
    // Language modeling specific buffers
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    float* d_embeddings_v;      // vocab_size x embed_dim
    
    // Working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float** d_layer_outputs;    // Array of layer outputs: seq_len x batch_size x vocab_size
    float** d_layer_residuals;  // Array of layer residuals: seq_len x batch_size x vocab_size
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float* d_losses;            // seq_len x batch_size
    
    // Dimensions
    int vocab_size;
    int embed_dim;
} SLM;

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

// CUDA kernel for residual connection: output = input + residual
__global__ void residual_add_kernel(float* output, float* input, float* residual, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + residual[idx];
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

// Initialize SLM with n layers
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int batch_size, int num_layers) {
    if (num_layers < 1) {
        printf("Error: num_layers must be at least 1\n");
        return NULL;
    }
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Set dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    slm->num_layers = num_layers;

    // Allocate arrays for layers
    slm->ssm_layers = (SSM**)malloc(num_layers * sizeof(SSM*));
    slm->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    slm->d_layer_outputs = (float**)malloc(num_layers * sizeof(float*));
    slm->d_layer_residuals = (float**)malloc(num_layers * sizeof(float*));

    // Initialize layers
    for (int layer = 0; layer < num_layers; layer++) {
        // First layer takes embed_dim input, others take vocab_size
        int input_dim = (layer == 0) ? embed_dim : slm->vocab_size;
        
        slm->ssm_layers[layer] = init_ssm(input_dim, state_dim, slm->vocab_size, seq_len, batch_size);
        slm->mlp_layers[layer] = init_mlp(slm->vocab_size, 4 * slm->vocab_size, slm->vocab_size, seq_len * batch_size);
        
        // Allocate layer output and residual buffers
        CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[layer], seq_len * batch_size * slm->vocab_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slm->d_layer_residuals[layer], seq_len * batch_size * slm->vocab_size * sizeof(float)));
    }

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
        // Free layers
        for (int layer = 0; layer < slm->num_layers; layer++) {
            if (slm->ssm_layers[layer]) free_ssm(slm->ssm_layers[layer]);
            if (slm->mlp_layers[layer]) free_mlp(slm->mlp_layers[layer]);
            cudaFree(slm->d_layer_outputs[layer]);
            cudaFree(slm->d_layer_residuals[layer]);
        }
        free(slm->ssm_layers);
        free(slm->mlp_layers);
        free(slm->d_layer_outputs);
        free(slm->d_layer_residuals);
        
        // Free other buffers
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

// Forward pass through n layers
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm_layers[0]->seq_len;
    int batch_size = slm->ssm_layers[0]->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // Current input starts with embeddings
    float* current_input = slm->d_embedded_input;
    
    // Process through all layers
    for (int layer = 0; layer < slm->num_layers; layer++) {
        // Forward through SSM
        // H_t = Input_t B^T + H_{t-1} A^T
        // O_t = H_t σ(H_t)  
        // Y_t = O_t C^T + Input_t D^T
        forward_pass_ssm(slm->ssm_layers[layer], current_input);

        // Forward through MLP
        // Z_t = Y_t W_1
        // A_t = Z_t σ(Z_t)
        // L_t = A_t W_2
        forward_pass_mlp(slm->mlp_layers[layer], slm->ssm_layers[layer]->d_predictions);
        
        // Store layer output
        CHECK_CUDA(cudaMemcpy(slm->d_layer_outputs[layer], slm->mlp_layers[layer]->d_predictions, 
                             total_tokens * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Residual connection: R_t = L_t + Y_t (skip connection from SSM to next layer)
        int residual_size = total_tokens * slm->vocab_size;
        int residual_blocks = (residual_size + 255) / 256;
        residual_add_kernel<<<residual_blocks, 256>>>(
            slm->d_layer_residuals[layer], 
            slm->d_layer_outputs[layer], 
            slm->ssm_layers[layer]->d_predictions, 
            residual_size
        );
        
        // Next layer's input is current layer's residual output
        current_input = slm->d_layer_residuals[layer];
    }
    
    // Final output is from last layer's MLP
    // P_t = softmax(L_final_t) - Apply softmax for probability distribution
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp_layers[slm->num_layers - 1]->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss: L = -1/(T·B) Σ_t Σ_b log P_{t,b,y_{t,b}}
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm_layers[0]->seq_len;
    int batch_size = slm->ssm_layers[0]->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // ∂L/∂L_final_t = P_t - 1_{y_t} - Compute cross-entropy gradient for final layer
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp_layers[slm->num_layers - 1]->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
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

// Zero gradients for all layers
void zero_gradients_slm(SLM* slm) {
    for (int layer = 0; layer < slm->num_layers; layer++) {
        zero_gradients_ssm(slm->ssm_layers[layer]);
        zero_gradients_mlp(slm->mlp_layers[layer]);
    }
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, 
                         slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass through n layers
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    int seq_len = slm->ssm_layers[0]->seq_len;
    int batch_size = slm->ssm_layers[0]->batch_size;
    int total_tokens = seq_len * batch_size;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Allocate temporary buffer for layer gradients
    float* d_layer_grad;
    CHECK_CUDA(cudaMalloc(&d_layer_grad, total_tokens * slm->vocab_size * sizeof(float)));
    
    // Start from the last layer and work backwards
    for (int layer = slm->num_layers - 1; layer >= 0; layer--) {
        // For the last layer, error is already set from loss calculation
        // For other layers, we need to propagate gradients
        
        if (layer < slm->num_layers - 1) {
            // Copy accumulated gradients to current layer's MLP error
            CHECK_CUDA(cudaMemcpy(slm->mlp_layers[layer]->d_error, d_layer_grad, 
                                 total_tokens * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        // Backward through MLP
        forward_pass_mlp(slm->mlp_layers[layer], slm->ssm_layers[layer]->d_predictions);
        backward_pass_mlp(slm->mlp_layers[layer], slm->ssm_layers[layer]->d_predictions);
        
        // Copy MLP input gradients to SSM error and prepare for residual
        CHECK_CUDA(cudaMemcpy(slm->ssm_layers[layer]->d_error, slm->mlp_layers[layer]->d_error, 
                             total_tokens * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Get input for this layer
        float* layer_input;
        if (layer == 0) {
            layer_input = slm->d_embedded_input;
        } else {
            layer_input = slm->d_layer_residuals[layer - 1];
        }
        
        // Backward through SSM
        backward_pass_ssm(slm->ssm_layers[layer], layer_input);
        
        // For layers before the last, compute gradients for previous layer
        if (layer > 0) {
            // Compute gradients for residual connection input
            // ∂L/∂R_{layer-1}_t = (∂L/∂H_layer_t) B_layer + (∂L/∂Y_layer_t) D_layer + ∂L/∂L_layer_t
            CHECK_CUDA(cudaMemset(d_layer_grad, 0, total_tokens * slm->vocab_size * sizeof(float)));
            
            for (int t = 0; t < seq_len; t++) {
                float* d_grad_t = d_layer_grad + t * batch_size * slm->vocab_size;
                float* d_state_error_t = slm->ssm_layers[layer]->d_state_error + t * batch_size * slm->ssm_layers[layer]->state_dim;
                float* d_output_error_t = slm->ssm_layers[layer]->d_error + t * batch_size * slm->vocab_size;
                
                // ∂L/∂R_{layer-1}_t += B_layer^T (∂L/∂H_layer_t)
                CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        slm->vocab_size, batch_size, slm->ssm_layers[layer]->state_dim,
                                        &alpha, slm->ssm_layers[layer]->d_B, slm->vocab_size,
                                        d_state_error_t, slm->ssm_layers[layer]->state_dim,
                                        &beta, d_grad_t, slm->vocab_size));
                
                // ∂L/∂R_{layer-1}_t += D_layer^T (∂L/∂Y_layer_t)
                CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        slm->vocab_size, batch_size, slm->vocab_size,
                                        &alpha, slm->ssm_layers[layer]->d_D, slm->vocab_size,
                                        d_output_error_t, slm->vocab_size,
                                        &alpha, d_grad_t, slm->vocab_size));
            }
            
            // Add MLP gradient (residual connection backward)
            CHECK_CUBLAS(cublasSaxpy(slm->ssm_layers[layer]->cublas_handle, total_tokens * slm->vocab_size,
                                    &alpha, slm->mlp_layers[layer]->d_error, 1, d_layer_grad, 1));
        }
    }
    
    // Compute embedding gradients from first layer
    // ∂L/∂E_t = (∂L/∂H_0_t) B_0 + (∂L/∂Y_0_t) D_0
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         seq_len * batch_size * slm->embed_dim * sizeof(float)));
    
    for (int t = 0; t < seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * batch_size * slm->embed_dim;
        float* d_state_error_t = slm->ssm_layers[0]->d_state_error + t * batch_size * slm->ssm_layers[0]->state_dim;
        float* d_output_error_t = slm->ssm_layers[0]->d_error + t * batch_size * slm->ssm_layers[0]->output_dim;
        
        // ∂L/∂E_t += B_0^T (∂L/∂H_0_t)
        CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[0]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm_layers[0]->state_dim,
                                &alpha, slm->ssm_layers[0]->d_B, slm->embed_dim,
                                d_state_error_t, slm->ssm_layers[0]->state_dim,
                                &beta, d_input_grad_t, slm->embed_dim));
        
        // ∂L/∂E_t += D_0^T (∂L/∂Y_0_t)
        CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[0]->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssm_layers[0]->output_dim,
                                &alpha, slm->ssm_layers[0]->d_D, slm->embed_dim,
                                d_output_error_t, slm->ssm_layers[0]->output_dim,
                                &alpha, d_input_grad_t, slm->embed_dim));
    }
    
    // ∂L/∂W_E[c] = Σ_{t,b: X_{t,b}=c} ∂L/∂E_t - Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        batch_size, slm->embed_dim
    );
    
    // Cleanup temporary buffer
    cudaFree(d_layer_grad);
}

// Update weights for all layers using AdamW
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update all SSM and MLP layers
    for (int layer = 0; layer < slm->num_layers; layer++) {
        update_weights_ssm(slm->ssm_layers[layer], learning_rate);
        update_weights_mlp(slm->mlp_layers[layer], learning_rate);
    }
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssm_layers[0]->beta1, slm->ssm_layers[0]->t);
    float beta2_t = powf(slm->ssm_layers[0]->beta2, slm->ssm_layers[0]->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g², W = (1-λη)W - η·m̂/√v̂
    adamw_update_kernel_ssm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssm_layers[0]->beta1, slm->ssm_layers[0]->beta2, slm->ssm_layers[0]->epsilon,
        learning_rate, slm->ssm_layers[0]->weight_decay, alpha_t,
        embed_size, slm->ssm_layers[0]->batch_size
    );
}

// Save model with all layers
void save_slm(SLM* slm, const char* filename) {
    // Save number of layers info file
    char info_file[256];
    strcpy(info_file, filename);
    char* dot = strrchr(info_file, '.');
    if (dot) *dot = '\0';
    strcat(info_file, "_info.txt");
    
    FILE* info_f = fopen(info_file, "w");
    if (info_f) {
        fprintf(info_f, "%d\n", slm->num_layers);
        fclose(info_f);
    }
    
    // Save first SSM layer with original filename
    save_ssm(slm->ssm_layers[0], filename);
    
    // Save other SSM layers
    for (int layer = 1; layer < slm->num_layers; layer++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        sprintf(ssm_file + strlen(ssm_file), "_ssm%d.bin", layer);
        save_ssm(slm->ssm_layers[layer], ssm_file);
    }
    
    // Save all MLP layers
    for (int layer = 0; layer < slm->num_layers; layer++) {
        char mlp_file[256];
        strcpy(mlp_file, filename);
        dot = strrchr(mlp_file, '.');
        if (dot) *dot = '\0';
        sprintf(mlp_file + strlen(mlp_file), "_mlp%d.bin", layer);
        save_mlp(slm->mlp_layers[layer], mlp_file);
    }
    
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
    printf("Model with %d layers saved\n", slm->num_layers);
}

// Load model with all layers
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load number of layers
    char info_file[256];
    strcpy(info_file, filename);
    char* dot = strrchr(info_file, '.');
    if (dot) *dot = '\0';
    strcat(info_file, "_info.txt");
    
    int num_layers = 1; // Default to 1 layer for backward compatibility
    FILE* info_f = fopen(info_file, "r");
    if (info_f) {
        fscanf(info_f, "%d", &num_layers);
        fclose(info_f);
    }
    
    // Load first SSM layer
    SSM* ssm0 = load_ssm(filename, custom_batch_size);
    if (!ssm0) return NULL;
    
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    slm->num_layers = num_layers;
    slm->vocab_size = ssm0->output_dim;
    slm->embed_dim = ssm0->input_dim;
    
    // Allocate layer arrays
    slm->ssm_layers = (SSM**)malloc(num_layers * sizeof(SSM*));
    slm->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    slm->d_layer_outputs = (float**)malloc(num_layers * sizeof(float*));
    slm->d_layer_residuals = (float**)malloc(num_layers * sizeof(float*));
    
    slm->ssm_layers[0] = ssm0;
    
    // Load other SSM layers
    for (int layer = 1; layer < num_layers; layer++) {
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        sprintf(ssm_file + strlen(ssm_file), "_ssm%d.bin", layer);
        slm->ssm_layers[layer] = load_ssm(ssm_file, custom_batch_size);
        
        if (!slm->ssm_layers[layer]) {
            // Cleanup on failure
            for (int i = 0; i < layer; i++) {
                free_ssm(slm->ssm_layers[i]);
            }
            free(slm->ssm_layers);
            free(slm->mlp_layers);
            free(slm->d_layer_outputs);
            free(slm->d_layer_residuals);
            free(slm);
            return NULL;
        }
    }
    
    // Load all MLP layers
    for (int layer = 0; layer < num_layers; layer++) {
        char mlp_file[256];
        strcpy(mlp_file, filename);
        dot = strrchr(mlp_file, '.');
        if (dot) *dot = '\0';
        sprintf(mlp_file + strlen(mlp_file), "_mlp%d.bin", layer);
        slm->mlp_layers[layer] = load_mlp(mlp_file, ssm0->seq_len * ssm0->batch_size);
        
        // Allocate layer buffers
        CHECK_CUDA(cudaMalloc(&slm->d_layer_outputs[layer], ssm0->seq_len * ssm0->batch_size * slm->vocab_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&slm->d_layer_residuals[layer], ssm0->seq_len * ssm0->batch_size * slm->vocab_size * sizeof(float)));
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, ssm0->seq_len * ssm0->batch_size * ssm0->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, ssm0->seq_len * ssm0->batch_size * ssm0->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, ssm0->seq_len * ssm0->batch_size * ssm0->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, ssm0->seq_len * ssm0->batch_size * sizeof(float)));
    
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
    
    printf("Model with %d layers loaded\n", num_layers);
    return slm;
}

// Text generation function for n layers
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
    
    // Allocate state buffers for all layers
    float** d_h_current = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_h_next = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_o_current = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_y_current = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_mlp_hidden = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_mlp_output = (float**)malloc(slm->num_layers * sizeof(float*));
    float** d_residual = (float**)malloc(slm->num_layers * sizeof(float*));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    
    // Allocate buffers for each layer
    for (int layer = 0; layer < slm->num_layers; layer++) {
        CHECK_CUDA(cudaMalloc(&d_h_current[layer], slm->ssm_layers[layer]->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_h_next[layer], slm->ssm_layers[layer]->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_o_current[layer], slm->ssm_layers[layer]->state_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y_current[layer], slm->vocab_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mlp_hidden[layer], slm->mlp_layers[layer]->hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_mlp_output[layer], slm->vocab_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_residual[layer], slm->vocab_size * sizeof(float)));
        
        // Initialize hidden states to zero
        CHECK_CUDA(cudaMemset(d_h_current[layer], 0, slm->ssm_layers[layer]->state_dim * sizeof(float)));
    }
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text + generation
    for (int i = 0; i < seed_len + generation_length; i++) {
        // Set input character
        if (i < seed_len) {
            h_input[0] = (unsigned char)seed_text[i];
        }
        // For generation, h_input[0] is set at the end of the previous iteration
        
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // E_t = W_E[X_t] - Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass through all layers
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        float* current_input = slm->d_embedded_input;
        
        for (int layer = 0; layer < slm->num_layers; layer++) {
            // SSM forward
            // H_t = Input_t B^T + H_{t-1} A^T
            CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    slm->ssm_layers[layer]->state_dim, 1, slm->ssm_layers[layer]->input_dim,
                                    &alpha, slm->ssm_layers[layer]->d_B, slm->ssm_layers[layer]->input_dim,
                                    current_input, slm->ssm_layers[layer]->input_dim,
                                    &beta, d_h_next[layer], slm->ssm_layers[layer]->state_dim));
            
            if (i > 0) {
                CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        slm->ssm_layers[layer]->state_dim, 1, slm->ssm_layers[layer]->state_dim,
                                        &alpha, slm->ssm_layers[layer]->d_A, slm->ssm_layers[layer]->state_dim,
                                        d_h_current[layer], slm->ssm_layers[layer]->state_dim,
                                        &beta_add, d_h_next[layer], slm->ssm_layers[layer]->state_dim));
            }
            
            // O_t = H_t σ(H_t)
            int block_size = 256;
            int num_blocks = (slm->ssm_layers[layer]->state_dim + block_size - 1) / block_size;
            swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_current[layer], d_h_next[layer], slm->ssm_layers[layer]->state_dim);
            
            // Y_t = O_t C^T + Input_t D^T
            CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    slm->ssm_layers[layer]->output_dim, 1, slm->ssm_layers[layer]->state_dim,
                                    &alpha, slm->ssm_layers[layer]->d_C, slm->ssm_layers[layer]->state_dim,
                                    d_o_current[layer], slm->ssm_layers[layer]->state_dim,
                                    &beta, d_y_current[layer], slm->ssm_layers[layer]->output_dim));
            
            CHECK_CUBLAS(cublasSgemm(slm->ssm_layers[layer]->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    slm->ssm_layers[layer]->output_dim, 1, slm->ssm_layers[layer]->input_dim,
                                    &alpha, slm->ssm_layers[layer]->d_D, slm->ssm_layers[layer]->input_dim,
                                    current_input, slm->ssm_layers[layer]->input_dim,
                                    &beta_add, d_y_current[layer], slm->ssm_layers[layer]->output_dim));
            
            // MLP forward
            CHECK_CUBLAS(cublasSgemv(slm->mlp_layers[layer]->cublas_handle,
                                    CUBLAS_OP_T,
                                    slm->mlp_layers[layer]->input_dim,
                                    slm->mlp_layers[layer]->hidden_dim,
                                    &alpha,
                                    slm->mlp_layers[layer]->d_fc1_weight,
                                    slm->mlp_layers[layer]->input_dim,
                                    d_y_current[layer],
                                    1,
                                    &beta,
                                    d_mlp_hidden[layer],
                                    1));

            num_blocks = (slm->mlp_layers[layer]->hidden_dim + block_size - 1) / block_size;
            swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
                d_mlp_hidden[layer],
                d_mlp_hidden[layer],
                slm->mlp_layers[layer]->hidden_dim
            );

            CHECK_CUBLAS(cublasSgemv(slm->mlp_layers[layer]->cublas_handle,
                                    CUBLAS_OP_T,
                                    slm->mlp_layers[layer]->hidden_dim,
                                    slm->mlp_layers[layer]->output_dim,
                                    &alpha,
                                    slm->mlp_layers[layer]->d_fc2_weight,
                                    slm->mlp_layers[layer]->hidden_dim,
                                    d_mlp_hidden[layer],
                                    1,
                                    &beta,
                                    d_mlp_output[layer],
                                    1));
            
            // Residual connection: R_t = L_t + Y_t
            residual_add_kernel<<<1, 256>>>(d_residual[layer], d_mlp_output[layer], d_y_current[layer], slm->vocab_size);
            
            // Next layer's input is current layer's residual output
            current_input = d_residual[layer];
            
            // Swap states for next iteration
            float* temp = d_h_current[layer];
            d_h_current[layer] = d_h_next[layer];
            d_h_next[layer] = temp;
        }
        
        // For generation (after seed), apply softmax and sample
        if (i >= seed_len) {
            // Apply softmax to final layer output
            softmax_kernel<<<1, 256>>>(slm->d_softmax, d_mlp_output[slm->num_layers - 1], 1, slm->vocab_size);
            
            // Copy probabilities to host and sample
            CHECK_CUDA(cudaMemcpy(h_probs, slm->d_softmax, slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Apply temperature scaling
            if (temperature != 1.0f) {
                float sum = 0.0f;
                for (int j = 0; j < slm->vocab_size; j++) {
                    h_probs[j] = expf(logf(h_probs[j] + 1e-15f) / temperature);
                    sum += h_probs[j];
                }
                for (int j = 0; j < slm->vocab_size; j++) {
                    h_probs[j] /= sum;
                }
            }
            
            // Sample from distribution
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
            
            // Ensure printable character
            if (sampled_char < 32 || sampled_char > 126) {
                sampled_char = 32; // space
            }
            
            printf("%c", sampled_char);
            fflush(stdout);
            
            h_input[0] = (unsigned char)sampled_char;
        }
    }
    
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_probs);
    cudaFree(d_input);
    
    for (int layer = 0; layer < slm->num_layers; layer++) {
        cudaFree(d_h_current[layer]);
        cudaFree(d_h_next[layer]);
        cudaFree(d_o_current[layer]);
        cudaFree(d_y_current[layer]);
        cudaFree(d_mlp_hidden[layer]);
        cudaFree(d_mlp_output[layer]);
        cudaFree(d_residual[layer]);
    }
    
    free(d_h_current);
    free(d_h_next);
    free(d_o_current);
    free(d_y_current);
    free(d_mlp_hidden);
    free(d_mlp_output);
    free(d_residual);
}

#endif