#ifndef SLM_H
#define SLM_H

#include "ssm/gpu/ssm.h"
#include "mlp/gpu/mlp.h"

typedef struct {
    SSM* ssm;                   // Underlying state space model
    MLP* mlp;                   // Multi-layer perceptron for output mapping
    
    // Language modeling specific buffers
    float* d_embeddings;        // vocab_size x embed_dim
    float* d_embeddings_grad;   // vocab_size x embed_dim
    float* d_embeddings_m;      // vocab_size x embed_dim
    float* d_embeddings_v;      // vocab_size x embed_dim
    
    // Working buffers
    float* d_embedded_input;    // seq_len x batch_size x embed_dim
    float* d_softmax;           // seq_len x batch_size x vocab_size
    float* d_input_gradients;   // seq_len x batch_size x embed_dim
    float* d_losses;            // seq_len x batch_size
    
    // Dimensions
    int vocab_size;
    int embed_dim;
} SLM;

// CUDA kernel for embedding lookup
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

// CUDA kernel for softmax
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

// CUDA kernel for cross-entropy gradient
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

// CUDA kernel for cross-entropy loss calculation
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

// CUDA kernel for embedding gradient accumulation
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
    slm->ssm = init_ssm(embed_dim, state_dim, slm->vocab_size, seq_len, batch_size);
    
    // Initialize MLP
    slm->mlp = init_mlp(slm->vocab_size, 4 * slm->vocab_size, slm->vocab_size, seq_len * batch_size);

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
    
    // Initialize embeddings
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
    
    // Embed characters
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256, seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, batch_size, slm->embed_dim
    );
    
    // Forward through SSM
    forward_pass_ssm(slm->ssm, slm->d_embedded_input);

    // Forward through MLP
    forward_pass_mlp(slm->mlp, slm->ssm->d_predictions);
    
    // Apply softmax
    int total_tokens = seq_len * batch_size;
    int blocks = (total_tokens + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax, slm->mlp->d_predictions, total_tokens, slm->vocab_size
    );
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    int seq_len = slm->ssm->seq_len;
    int batch_size = slm->ssm->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // Compute cross-entropy gradient (softmax - one_hot) for backprop
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_gradient_kernel<<<blocks, 256>>>(
        slm->mlp->d_error, slm->d_softmax, d_y, total_tokens, slm->vocab_size
    );
    
    // Calculate actual cross-entropy loss: -log(softmax[target])
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
    // Backward through MLP
    backward_pass_mlp(slm->mlp, slm->ssm->d_predictions);
    
    // Copy MLP input gradients to SSM
    int total_elements = slm->ssm->seq_len * slm->ssm->batch_size * slm->ssm->output_dim;
    CHECK_CUDA(cudaMemcpy(slm->ssm->d_error, slm->mlp->d_error, 
                         total_elements * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Backward through SSM
    backward_pass_ssm(slm->ssm, slm->d_embedded_input);
    
    // Compute input gradients
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CHECK_CUDA(cudaMemset(slm->d_input_gradients, 0, 
                         slm->ssm->seq_len * slm->ssm->batch_size * slm->ssm->input_dim * sizeof(float)));
    
    for (int t = 0; t < slm->ssm->seq_len; t++) {
        float* d_input_grad_t = slm->d_input_gradients + t * slm->ssm->batch_size * slm->ssm->input_dim;
        float* d_state_error_t = slm->ssm->d_state_error + t * slm->ssm->batch_size * slm->ssm->state_dim;
        float* d_output_error_t = slm->ssm->d_error + t * slm->ssm->batch_size * slm->ssm->output_dim;
        
        // Gradient from state path
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm->input_dim, slm->ssm->batch_size, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                d_state_error_t, slm->ssm->state_dim,
                                &beta, d_input_grad_t, slm->ssm->input_dim));
        
        // Gradient from output path
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->ssm->input_dim, slm->ssm->batch_size, slm->ssm->output_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                d_output_error_t, slm->ssm->output_dim,
                                &alpha, d_input_grad_t, slm->ssm->input_dim));
    }
    
    // Accumulate embedding gradients
    dim3 block(256);
    dim3 grid((slm->ssm->batch_size + 255) / 256, slm->ssm->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        slm->ssm->batch_size, slm->embed_dim
    );
}

// Update weights
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update SSM weights
    update_weights_ssm(slm->ssm, learning_rate);
    
    // Update MLP weights
    update_weights_mlp(slm->mlp, learning_rate);
    
    // Update embeddings
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t = powf(slm->ssm->beta1, slm->ssm->t);
    float beta2_t = powf(slm->ssm->beta2, slm->ssm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
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

void forward_pass_mlp_single(MLP* mlp, float* d_input, float* d_output);

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
    float* d_y_current = NULL;
    float* d_mlp_input = NULL;
    float* d_mlp_output = NULL;
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_h_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h_next, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_o_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_current, slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_input, slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mlp_output, slm->vocab_size * sizeof(float)));
    
    // Initialize hidden state to zero
    CHECK_CUDA(cudaMemset(d_h_current, 0, slm->ssm->state_dim * sizeof(float)));
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text to build up hidden state
    for (int i = 0; i < seed_len; i++) {
        h_input[0] = (unsigned char)seed_text[i];
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = X_t B^T
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
        
        // Y_t = O_t C^T + X_t D^T
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_C, slm->ssm->state_dim,
                                d_o_current, slm->ssm->state_dim,
                                &beta, d_mlp_input, slm->ssm->output_dim));
        
        // Y_t += X_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta_add, d_mlp_input, slm->ssm->output_dim));
        
        // Forward through MLP (single sample)
        forward_pass_mlp_single(slm->mlp, d_mlp_input, d_mlp_output);
        
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
        
        // Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = X_t B^T + H_{t-1} A^T
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
        
        // Y_t = O_t C^T + X_t D^T
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_C, slm->ssm->state_dim,
                                d_o_current, slm->ssm->state_dim,
                                &beta, d_mlp_input, slm->ssm->output_dim));
        
        // Y_t += X_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta_add, d_mlp_input, slm->ssm->output_dim));
        
        // Forward through MLP (single sample)
        forward_pass_mlp_single(slm->mlp, d_mlp_input, d_mlp_output);
        
        // Apply softmax
        softmax_kernel<<<1, 256>>>(slm->d_softmax, d_mlp_output, 1, slm->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_probs, slm->d_softmax, slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling
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
    cudaFree(d_y_current);
    cudaFree(d_mlp_input);
    cudaFree(d_mlp_output);
}

// Helper function for single sample MLP forward pass
void forward_pass_mlp_single(MLP* mlp, float* d_input, float* d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Layer 1: d_input -> d_layer1_output
    CHECK_CUBLAS(cublasSgemv(mlp->cublas_handle,
                            CUBLAS_OP_T,
                            mlp->input_dim,
                            mlp->hidden_dim,
                            &alpha,
                            mlp->d_fc1_weight,
                            mlp->input_dim,
                            d_input,
                            1,
                            &beta,
                            mlp->d_layer1_output,
                            1));

    // Apply swish activation
    int block_size = 256;
    int num_blocks = (mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_layer1_output,
        mlp->d_layer1_output,
        mlp->hidden_dim
    );

    // Layer 2: d_layer1_output -> d_output
    CHECK_CUBLAS(cublasSgemv(mlp->cublas_handle,
                            CUBLAS_OP_T,
                            mlp->hidden_dim,
                            mlp->output_dim,
                            &alpha,
                            mlp->d_fc2_weight,
                            mlp->hidden_dim,
                            mlp->d_layer1_output,
                            1,
                            &beta,
                            d_output,
                            1));
}

#endif