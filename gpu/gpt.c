#include "gpt.h"

// Initialize the GPT
GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    GPT* gpt = (GPT*)malloc(sizeof(GPT));
    
    // Store dimensions
    gpt->seq_len = seq_len;
    gpt->d_model = d_model;
    gpt->batch_size = batch_size;
    gpt->hidden_dim = hidden_dim;
    gpt->num_layers = num_layers;
    gpt->vocab_size = 65536;
    gpt->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    gpt->beta1 = 0.9f;
    gpt->beta2 = 0.999f;
    gpt->epsilon = 1e-8f;
    gpt->t = 0;
    gpt->weight_decay = 0.01f;
    
    int token_emb_size = gpt->vocab_size * d_model;
    int output_weight_size = d_model * gpt->vocab_size;
    int embedded_size = batch_size * seq_len * d_model;
    int output_size = batch_size * seq_len * gpt->vocab_size;
    
    // Allocate host memory for embedding initialization
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_W_output = (float*)malloc(output_weight_size * sizeof(float));
    
    // Initialize token embeddings on host
    float token_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < token_emb_size; i++) {
        h_token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Initialize output weights on host
    for (int i = 0; i < output_weight_size; i++) {
        h_W_output[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Allocate device memory for embeddings and gradients
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_grad, token_emb_size * sizeof(float)));
    
    // Allocate device memory for output weights and gradients
    CHECK_CUDA(cudaMalloc(&gpt->d_W_output, output_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_W_output_grad, output_weight_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_token_embedding_v, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_W_output_m, output_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_W_output_v, output_weight_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&gpt->d_embedded_input, embedded_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gpt->d_output, output_size * sizeof(float)));
    
    // Alias device memory for backward pass buffers
    gpt->d_grad_output = gpt->d_output;
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&gpt->d_loss_result, sizeof(float)));
    
    // Copy embeddings and weights to device
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_W_output, h_W_output, output_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_W_output_m, 0, output_weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_W_output_v, 0, output_weight_size * sizeof(float)));
    
    // Initialize transformer
    gpt->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, true, cublaslt_handle);
    
    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&gpt->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // Output weight: [d_model x vocab_size]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->output_weight_layout, CUDA_R_32F, d_model, gpt->vocab_size, gpt->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->output_weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Flattened sequence data (d_model): [batch_size * seq_len x d_model]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->seq_flat_d_model_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->seq_flat_d_model_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Flattened sequence data (vocab): [batch_size * seq_len x vocab_size]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&gpt->seq_flat_vocab_layout, CUDA_R_32F, batch_size * seq_len, gpt->vocab_size, gpt->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(gpt->seq_flat_vocab_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_token_embedding);
    free(h_W_output);
    
    return gpt;
}

// Free GPT memory
void free_gpt(GPT* gpt) {
    // Free transformer
    free_transformer(gpt->transformer);
    
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(gpt->matmul_desc);
    
    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(gpt->output_weight_layout);
    cublasLtMatrixLayoutDestroy(gpt->seq_flat_d_model_layout);
    cublasLtMatrixLayoutDestroy(gpt->seq_flat_vocab_layout);
    
    // Free device memory
    cudaFree(gpt->d_token_embedding); cudaFree(gpt->d_token_embedding_grad);
    cudaFree(gpt->d_token_embedding_m); cudaFree(gpt->d_token_embedding_v);
    cudaFree(gpt->d_W_output); cudaFree(gpt->d_W_output_grad);
    cudaFree(gpt->d_W_output_m); cudaFree(gpt->d_W_output_v);
    cudaFree(gpt->d_embedded_input);
    cudaFree(gpt->d_output);
    
    // Free loss computation buffer
    cudaFree(gpt->d_loss_result);
    
    free(gpt);
}

// CUDA kernel for token embedding lookup
__global__ static void token_embedding_lookup_kernel(float* embedded, float* token_embedding, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
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
__global__ static void softmax_cross_entropy_kernel(float* loss_result, float* grad_logits, float* logits, unsigned short* targets, int batch_size, int seq_len, int vocab_size) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= seq_len) return;
    
    int logits_offset = b * seq_len * vocab_size + t * vocab_size;
    float* logits_bt = &logits[logits_offset];
    float* grad_logits_bt = &grad_logits[logits_offset];
    
    // Find max for numerical stability
    float max_logit = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        if (logits_bt[v] > max_logit) max_logit = logits_bt[v];
    }
    
    // Compute softmax probabilities
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        float exp_val = expf(logits_bt[v] - max_logit);
        grad_logits_bt[v] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize to get probabilities
    for (int v = 0; v < vocab_size; v++) {
        grad_logits_bt[v] /= sum_exp;
    }
    
    // Compute cross-entropy loss and gradient
    int target_token = targets[b * seq_len + t];
    float target_prob = grad_logits_bt[target_token];
    
    // Add cross-entropy loss
    atomicAdd(loss_result, -logf(target_prob + 1e-10f));
    
    // Set gradient: softmax - one_hot
    grad_logits_bt[target_token] -= 1.0f;
}

// CUDA kernel for token embedding gradient accumulation
__global__ static void token_embedding_grad_kernel(float* token_embedding_grad, float* grad_embedded, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
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
void forward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Token embedding lookup
    dim3 grid_emb(gpt->batch_size, gpt->seq_len);
    dim3 block_emb(gpt->d_model);
    token_embedding_lookup_kernel<<<grid_emb, block_emb>>>(
        gpt->d_embedded_input, gpt->d_token_embedding, d_input_tokens,
        gpt->batch_size, gpt->seq_len, gpt->d_model
    );
    
    // Step 2: Forward pass through transformer
    forward_pass_transformer(gpt->transformer, gpt->d_embedded_input);
    
    // Step 3: Output projection: output = transformer_output * W_output
    LT_MATMUL(gpt, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              gpt->transformer->mlp_layers[gpt->num_layers-1]->d_output, gpt->seq_flat_d_model_layout,
              gpt->d_W_output, gpt->output_weight_layout,
              &beta, gpt->d_output, gpt->seq_flat_vocab_layout);
}

// Calculate loss
float calculate_loss_gpt(GPT* gpt, unsigned short* d_target_tokens) {
    // Reset loss accumulator
    CHECK_CUDA(cudaMemset(gpt->d_loss_result, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    dim3 grid(gpt->batch_size, gpt->seq_len);
    softmax_cross_entropy_kernel<<<grid, 1>>>(
        gpt->d_loss_result, gpt->d_grad_output, gpt->d_output, d_target_tokens,
        gpt->batch_size, gpt->seq_len, gpt->vocab_size
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, gpt->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / (gpt->batch_size * gpt->seq_len);
}

// Zero gradients
void zero_gradients_gpt(GPT* gpt) {
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    int output_weight_size = gpt->d_model * gpt->vocab_size;
    
    CHECK_CUDA(cudaMemset(gpt->d_token_embedding_grad, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(gpt->d_W_output_grad, 0, output_weight_size * sizeof(float)));
    
    zero_gradients_transformer(gpt->transformer);
}

// Backward pass
void backward_pass_gpt(GPT* gpt, unsigned short* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 3 (backward): Backward pass through output projection
    // grad_W_output = transformer_output^T * grad_output
    LT_MATMUL(gpt, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              gpt->transformer->mlp_layers[gpt->num_layers-1]->d_output, gpt->seq_flat_d_model_layout,
              gpt->d_grad_output, gpt->seq_flat_vocab_layout,
              &alpha, gpt->d_W_output_grad, gpt->output_weight_layout);
    
    // grad_transformer_output = grad_output * W_output^T
    LT_MATMUL(gpt, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              gpt->d_grad_output, gpt->seq_flat_vocab_layout,
              gpt->d_W_output, gpt->output_weight_layout,
              &beta, gpt->transformer->mlp_layers[gpt->num_layers-1]->d_grad_output, gpt->seq_flat_d_model_layout);
    
    // Step 2 (backward): Backward pass through transformer
    backward_pass_transformer(gpt->transformer, gpt->d_embedded_input, gpt->d_embedded_input);
    
    // Step 1 (backward): Token embedding gradients
    dim3 grid_emb(gpt->batch_size, gpt->seq_len);
    dim3 block_emb(gpt->d_model);
    
    token_embedding_grad_kernel<<<grid_emb, block_emb>>>(
        gpt->d_token_embedding_grad, gpt->d_embedded_input, d_input_tokens,
        gpt->batch_size, gpt->seq_len, gpt->d_model
    );
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel_gpt(float* weight, float* grad, float* m, float* v,
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
void update_weights_gpt(GPT* gpt, float learning_rate, int effective_batch_size) {
    gpt->t++;
    
    float beta1_t = powf(gpt->beta1, gpt->t);
    float beta2_t = powf(gpt->beta2, gpt->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update token embeddings
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    int token_blocks = (token_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_gpt<<<token_blocks, block_size>>>(
        gpt->d_token_embedding, gpt->d_token_embedding_grad, gpt->d_token_embedding_m, gpt->d_token_embedding_v,
        gpt->beta1, gpt->beta2, gpt->epsilon, learning_rate, gpt->weight_decay,
        alpha_t, token_emb_size, effective_batch_size
    );
    
    // Update output weights
    int output_weight_size = gpt->d_model * gpt->vocab_size;
    int output_blocks = (output_weight_size + block_size - 1) / block_size;
    adamw_update_kernel_gpt<<<output_blocks, block_size>>>(
        gpt->d_W_output, gpt->d_W_output_grad, gpt->d_W_output_m, gpt->d_W_output_v,
        gpt->beta1, gpt->beta2, gpt->epsilon, learning_rate, gpt->weight_decay,
        alpha_t, output_weight_size, effective_batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(gpt->transformer, learning_rate, effective_batch_size);
}

// Serialize GPT to a file
static void serialize_gpt(GPT* gpt, FILE* file) {
    // Write dimensions
    fwrite(&gpt->seq_len, sizeof(int), 1, file);
    fwrite(&gpt->d_model, sizeof(int), 1, file);
    fwrite(&gpt->hidden_dim, sizeof(int), 1, file);
    fwrite(&gpt->num_layers, sizeof(int), 1, file);
    fwrite(&gpt->vocab_size, sizeof(int), 1, file);
    
    int token_emb_size = gpt->vocab_size * gpt->d_model;
    int output_weight_size = gpt->d_model * gpt->vocab_size;
    
    // Allocate host memory and copy embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_W_output = (float*)malloc(output_weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding, gpt->d_token_embedding, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_output, gpt->d_W_output, output_weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write embeddings and weights
    fwrite(h_token_embedding, sizeof(float), token_emb_size, file);
    fwrite(h_W_output, sizeof(float), output_weight_size, file);
    
    free(h_token_embedding);
    free(h_W_output);
    
    // Write optimizer state
    fwrite(&gpt->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    float* h_W_output_m = (float*)malloc(output_weight_size * sizeof(float));
    float* h_W_output_v = (float*)malloc(output_weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding_m, gpt->d_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_v, gpt->d_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_output_m, gpt->d_W_output_m, output_weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W_output_v, gpt->d_W_output_v, output_weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_v, sizeof(float), token_emb_size, file);
    fwrite(h_W_output_m, sizeof(float), output_weight_size, file);
    fwrite(h_W_output_v, sizeof(float), output_weight_size, file);
    
    free(h_token_embedding_m); free(h_token_embedding_v);
    free(h_W_output_m); free(h_W_output_v);
    
    // Serialize transformer
    serialize_transformer(gpt->transformer, file);
}

// Deserialize GPT from a file
static GPT* deserialize_gpt(FILE* file, int batch_size, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int seq_len, d_model, hidden_dim, num_layers, vocab_size;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    
    // Initialize GPT
    GPT* gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    int token_emb_size = vocab_size * d_model;
    int output_weight_size = d_model * vocab_size;
    
    // Load embeddings and weights
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_W_output = (float*)malloc(output_weight_size * sizeof(float));
    
    fread(h_token_embedding, sizeof(float), token_emb_size, file);
    fread(h_W_output, sizeof(float), output_weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_W_output, h_W_output, output_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_token_embedding);
    free(h_W_output);
    
    // Load optimizer state
    fread(&gpt->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    float* h_W_output_m = (float*)malloc(output_weight_size * sizeof(float));
    float* h_W_output_v = (float*)malloc(output_weight_size * sizeof(float));
    
    fread(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_v, sizeof(float), token_emb_size, file);
    fread(h_W_output_m, sizeof(float), output_weight_size, file);
    fread(h_W_output_v, sizeof(float), output_weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding_m, h_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_token_embedding_v, h_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_W_output_m, h_W_output_m, output_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gpt->d_W_output_v, h_W_output_v, output_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_token_embedding_m); free(h_token_embedding_v);
    free(h_W_output_m); free(h_W_output_v);
    
    // Free the initialized transformer
    free_transformer(gpt->transformer);
    
    // Deserialize transformer
    gpt->transformer = deserialize_transformer(file, batch_size, cublaslt_handle);
    
    return gpt;
}

// Save GPT to a file
void save_gpt(GPT* gpt, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    serialize_gpt(gpt, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load GPT from a file
GPT* load_gpt(const char* filename, int batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    GPT* gpt = deserialize_gpt(file, batch_size, cublaslt_handle);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return gpt;
}