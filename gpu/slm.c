#include "slm.h"

// Initialize the SLM
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Hard set vocab_size to 256
    slm->vocab_size = 256;
    
    // Ensure d_model equals vocab_size
    if (d_model != slm->vocab_size) {
        printf("Error: d_model (%d) must equal vocab_size (%d)\n", d_model, slm->vocab_size);
        free(slm);
        return NULL;
    }
    
    // Store dimensions
    slm->seq_len = seq_len;
    slm->d_model = d_model;
    slm->batch_size = batch_size;
    slm->hidden_dim = hidden_dim;
    slm->num_layers = num_layers;
    slm->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;
    
    int token_emb_size = slm->vocab_size * d_model;
    int embedded_size = batch_size * seq_len * d_model;
    
    // Allocate host memory for embedding initialization
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    
    // Initialize embeddings on host
    float token_scale = 1.0f / sqrtf(d_model);
    for (int i = 0; i < token_emb_size; i++) {
        h_token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Allocate device memory for embeddings and gradients
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_grad, token_emb_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_v, token_emb_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, embedded_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_grad_output, embedded_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_grad_embedded, embedded_size * sizeof(float)));
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&slm->d_loss_result, sizeof(float)));
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    
    // Initialize transformer
    slm->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, cublaslt_handle);
    
    // Set d_output to point to transformer output
    slm->d_output = slm->transformer->mlp_layers[num_layers-1]->d_layer_output;
    
    // Free host memory
    free(h_token_embedding);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    // Free transformer
    free_transformer(slm->transformer);
    
    // Free device memory
    cudaFree(slm->d_token_embedding);
    cudaFree(slm->d_token_embedding_grad);
    cudaFree(slm->d_token_embedding_m);
    cudaFree(slm->d_token_embedding_v);
    cudaFree(slm->d_embedded_input);
    cudaFree(slm->d_grad_output);
    cudaFree(slm->d_grad_embedded);
    cudaFree(slm->d_loss_result);
    
    free(slm);
}

// CUDA kernel for token embedding lookup
__global__ static void token_embedding_lookup_kernel(float* embedded, float* token_embedding, unsigned char* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int token_idx = b * seq_len + t;
    int token = tokens[token_idx];
    int emb_idx = b * seq_len * d_model + t * d_model + d;
    
    embedded[emb_idx] = token_embedding[token * d_model + d];
}

// CUDA kernel for softmax and cross-entropy loss computation
__global__ static void softmax_cross_entropy_kernel(float* loss_result, float* grad_output, float* logits, unsigned char* targets, int batch_size, int seq_len, int vocab_size) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;
    
    int logits_offset = b * seq_len * vocab_size + t * vocab_size;
    float* logits_bt = &logits[logits_offset];
    float* grad_output_bt = &grad_output[logits_offset];
    
    // Find max for numerical stability
    float max_logit = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        if (logits_bt[v] > max_logit) max_logit = logits_bt[v];
    }
    
    // Compute softmax probabilities
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        float exp_val = expf(logits_bt[v] - max_logit);
        grad_output_bt[v] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize to get probabilities
    for (int v = 0; v < vocab_size; v++) {
        grad_output_bt[v] /= sum_exp;
    }
    
    // Compute cross-entropy loss and gradient
    int target_token = targets[b * seq_len + t];
    float target_prob = grad_output_bt[target_token];
    
    // Add cross-entropy loss
    atomicAdd(loss_result, -logf(target_prob + 1e-10f));
    
    // Set gradient: softmax - one_hot
    grad_output_bt[target_token] -= 1.0f;
}

// CUDA kernel for token embedding gradient accumulation
__global__ static void token_embedding_grad_kernel(float* token_embedding_grad, float* grad_embedded, unsigned char* tokens, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int token_idx = b * seq_len + t;
    int token = tokens[token_idx];
    int emb_idx = b * seq_len * d_model + t * d_model + d;
    
    atomicAdd(&token_embedding_grad[token * d_model + d], grad_embedded[emb_idx]);
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    // Step 1: Token embedding lookup
    dim3 grid_emb(slm->batch_size, slm->seq_len);
    dim3 block_emb(slm->d_model);
    token_embedding_lookup_kernel<<<grid_emb, block_emb>>>(
        slm->d_embedded_input, slm->d_token_embedding, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model
    );
    
    // Step 2: Forward pass through transformer
    forward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // d_output now points to transformer output (logits)
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens) {
    // Reset loss accumulator
    CHECK_CUDA(cudaMemset(slm->d_loss_result, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    dim3 grid(slm->batch_size, slm->seq_len);
    softmax_cross_entropy_kernel<<<grid, 1>>>(
        slm->d_loss_result, slm->d_grad_output, slm->d_output, d_target_tokens,
        slm->batch_size, slm->seq_len, slm->vocab_size
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, slm->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / (slm->batch_size * slm->seq_len);
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    int token_emb_size = slm->vocab_size * slm->d_model;
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_grad, 0, token_emb_size * sizeof(float)));
    zero_gradients_transformer(slm->transformer);
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    // Step 1: Backward pass through transformer
    backward_pass_transformer(slm->transformer, slm->d_embedded_input, slm->d_grad_embedded);
    
    // Step 2: Gradient through token embeddings
    dim3 grid_emb(slm->batch_size, slm->seq_len);
    dim3 block_emb(slm->d_model);
    token_embedding_grad_kernel<<<grid_emb, block_emb>>>(
        slm->d_token_embedding_grad, slm->d_grad_embedded, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model
    );
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_slm(float* weight, float* grad, float* m, float* v,
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
    
    // Update token embeddings
    int token_emb_size = slm->vocab_size * slm->d_model;
    int token_blocks = (token_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<token_blocks, block_size>>>(
        slm->d_token_embedding, slm->d_token_embedding_grad, slm->d_token_embedding_m, slm->d_token_embedding_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, token_emb_size, slm->batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(slm->transformer, learning_rate);
}

// Save function
void save_slm(SLM* slm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&slm->seq_len, sizeof(int), 1, file);
    fwrite(&slm->d_model, sizeof(int), 1, file);
    fwrite(&slm->batch_size, sizeof(int), 1, file);
    fwrite(&slm->hidden_dim, sizeof(int), 1, file);
    fwrite(&slm->num_layers, sizeof(int), 1, file);
    fwrite(&slm->vocab_size, sizeof(int), 1, file);
    fwrite(&slm->transformer->attention_layers[0]->is_causal, sizeof(bool), 1, file);
    
    // Save Adam state
    fwrite(&slm->t, sizeof(int), 1, file);
    
    // Save token embeddings and Adam parameters
    int token_emb_size = slm->vocab_size * slm->d_model;
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding, slm->d_token_embedding, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_m, slm->d_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_v, slm->d_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    free(h_token_embedding);
    free(h_token_embedding_m);
    free(h_token_embedding_v);
    
    fclose(file);
    
    // Save transformer
    char transformer_filename[256];
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", filename);
    save_transformer(slm->transformer, transformer_filename);
    
    printf("Model saved to %s\n", filename);
}

// Load function
SLM* load_slm(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers, vocab_size;
    bool is_causal;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize SLM
    SLM* slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, cublaslt_handle);
    
    // Load Adam state
    fread(&slm->t, sizeof(int), 1, file);
    
    // Load token embeddings and Adam parameters
    int token_emb_size = vocab_size * d_model;
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    
    fread(h_token_embedding, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_v, sizeof(float), token_emb_size, file);
    
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding_m, h_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding_v, h_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_token_embedding);
    free(h_token_embedding_m);
    free(h_token_embedding_v);
    
    fclose(file);
    
    // Load transformer
    char transformer_filename[256];
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", filename);
    free_transformer(slm->transformer);
    slm->transformer = load_transformer(transformer_filename, batch_size, cublaslt_handle);
    slm->d_output = slm->transformer->mlp_layers[num_layers-1]->d_layer_output;
    
    printf("Model loaded from %s\n", filename);
    return slm;
}