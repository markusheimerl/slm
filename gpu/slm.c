#include "slm.h"

// Initialize the SLM
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, cublasLtHandle_t cublaslt_handle) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->seq_len = seq_len;
    slm->d_model = d_model;
    slm->batch_size = batch_size;
    slm->hidden_dim = hidden_dim;
    slm->num_layers = num_layers;
    slm->vocab_size = 256;
    slm->cublaslt_handle = cublaslt_handle;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;
    
    int token_emb_size = slm->vocab_size * d_model;
    int pos_emb_size = seq_len * d_model;
    int output_proj_size = d_model * slm->vocab_size;
    int embedded_size = batch_size * seq_len * d_model;
    int logits_size = batch_size * seq_len * slm->vocab_size;
    
    // Allocate host memory for embedding initialization
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_position_embedding = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_projection = (float*)malloc(output_proj_size * sizeof(float));
    
    // Initialize embeddings on host
    float token_scale = 1.0f / sqrtf(d_model);
    float pos_scale = 1.0f / sqrtf(d_model);
    float output_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < token_emb_size; i++) {
        h_token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    for (int i = 0; i < pos_emb_size; i++) {
        h_position_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * pos_scale;
    }
    
    for (int i = 0; i < output_proj_size; i++) {
        h_output_projection[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * output_scale;
    }
    
    // Allocate device memory for embeddings and gradients
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection, output_proj_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_grad, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_grad, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_grad, output_proj_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_m, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_token_embedding_v, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_m, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_position_embedding_v, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_m, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_output_projection_v, output_proj_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, embedded_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_logits, logits_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_probs, logits_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&slm->d_grad_embedded, embedded_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_grad_logits, logits_size * sizeof(float)));
    
    // Allocate single device float for loss computation
    CHECK_CUDA(cudaMalloc(&slm->d_loss_result, sizeof(float)));
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_position_embedding, h_position_embedding, pos_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_projection, h_output_projection, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_m, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_v, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_m, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_v, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_m, 0, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_v, 0, output_proj_size * sizeof(float)));
    
    // Initialize transformer
    slm->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, cublaslt_handle);
    
    // Create cuBLASLt descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&slm->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&slm->matmul_NT_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&slm->matmul_TN_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Set transpose operations
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    transA = CUBLAS_OP_T;
    transB = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(slm->matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layouts
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->token_emb_layout, CUDA_R_32F, slm->vocab_size, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->token_emb_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->pos_emb_layout, CUDA_R_32F, seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->pos_emb_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->output_proj_layout, CUDA_R_32F, d_model, slm->vocab_size, slm->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->output_proj_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->embedded_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->embedded_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->logits_layout, CUDA_R_32F, batch_size * seq_len, slm->vocab_size, slm->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->logits_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Special layouts for gradient computation
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->transformer_out_for_grad_layout, CUDA_R_32F, batch_size * seq_len, d_model, d_model));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->transformer_out_for_grad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&slm->grad_logits_for_grad_layout, CUDA_R_32F, batch_size * seq_len, slm->vocab_size, slm->vocab_size));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(slm->grad_logits_for_grad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_token_embedding); free(h_position_embedding); free(h_output_projection);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    // Free transformer
    free_transformer(slm->transformer);
    
    // Destroy cuBLASLt descriptors
    cublasLtMatmulDescDestroy(slm->matmul_desc);
    cublasLtMatmulDescDestroy(slm->matmul_NT_desc);
    cublasLtMatmulDescDestroy(slm->matmul_TN_desc);
    
    // Destroy layouts
    cublasLtMatrixLayoutDestroy(slm->token_emb_layout);
    cublasLtMatrixLayoutDestroy(slm->pos_emb_layout);
    cublasLtMatrixLayoutDestroy(slm->output_proj_layout);
    cublasLtMatrixLayoutDestroy(slm->embedded_layout);
    cublasLtMatrixLayoutDestroy(slm->logits_layout);
    cublasLtMatrixLayoutDestroy(slm->transformer_out_for_grad_layout);
    cublasLtMatrixLayoutDestroy(slm->grad_logits_for_grad_layout);
    
    // Free device memory
    cudaFree(slm->d_token_embedding); cudaFree(slm->d_position_embedding); cudaFree(slm->d_output_projection);
    cudaFree(slm->d_token_embedding_grad); cudaFree(slm->d_position_embedding_grad); cudaFree(slm->d_output_projection_grad);
    cudaFree(slm->d_token_embedding_m); cudaFree(slm->d_token_embedding_v);
    cudaFree(slm->d_position_embedding_m); cudaFree(slm->d_position_embedding_v);
    cudaFree(slm->d_output_projection_m); cudaFree(slm->d_output_projection_v);
    cudaFree(slm->d_embedded_input); cudaFree(slm->d_logits); cudaFree(slm->d_probs);
    cudaFree(slm->d_grad_embedded); cudaFree(slm->d_grad_logits);
    
    // Free loss computation buffer
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

// CUDA kernel for position embedding addition
__global__ static void position_embedding_add_kernel(float* embedded, float* position_embedding, int batch_size, int seq_len, int d_model) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    
    if (b >= batch_size || t >= seq_len || d >= d_model) return;
    
    int emb_idx = b * seq_len * d_model + t * d_model + d;
    embedded[emb_idx] += position_embedding[t * d_model + d];
}

// CUDA kernel for softmax and cross-entropy loss computation
__global__ static void softmax_cross_entropy_kernel(float* loss_result, float* grad_logits, float* logits, unsigned char* targets, int batch_size, int seq_len, int vocab_size) {
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

// CUDA kernel for position embedding gradient accumulation
__global__ static void position_embedding_grad_kernel(float* position_embedding_grad, float* grad_embedded, int batch_size, int seq_len, int d_model) {
    int t = blockIdx.x;
    int d = threadIdx.x;
    
    if (t >= seq_len || d >= d_model) return;
    
    float grad_sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int emb_idx = b * seq_len * d_model + t * d_model + d;
        grad_sum += grad_embedded[emb_idx];
    }
    
    position_embedding_grad[t * d_model + d] += grad_sum;
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 1: Token embedding lookup
    dim3 grid_emb(slm->batch_size, slm->seq_len);
    dim3 block_emb(slm->d_model);
    token_embedding_lookup_kernel<<<grid_emb, block_emb>>>(
        slm->d_embedded_input, slm->d_token_embedding, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model
    );
    
    // Step 2: Add position embeddings
    position_embedding_add_kernel<<<grid_emb, block_emb>>>(
        slm->d_embedded_input, slm->d_position_embedding,
        slm->batch_size, slm->seq_len, slm->d_model
    );
    
    // Step 3: Forward pass through transformer
    forward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // Step 4: Output projection to vocabulary
    CHECK_CUBLASLT(cublasLtMatmul(slm->cublaslt_handle,
                                  slm->matmul_desc,
                                  &alpha,
                                  slm->transformer->mlp_layers[slm->num_layers-1]->d_layer_output, slm->embedded_layout,
                                  slm->d_output_projection, slm->output_proj_layout,
                                  &beta,
                                  slm->d_logits, slm->logits_layout,
                                  slm->d_logits, slm->logits_layout,
                                  NULL, NULL, 0, 0));
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_target_tokens) {
    // Reset loss accumulator
    CHECK_CUDA(cudaMemset(slm->d_loss_result, 0, sizeof(float)));
    
    // Compute softmax and cross-entropy loss
    dim3 grid(slm->batch_size, slm->seq_len);
    softmax_cross_entropy_kernel<<<grid, 1>>>(
        slm->d_loss_result, slm->d_grad_logits, slm->d_logits, d_target_tokens,
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
    int pos_emb_size = slm->seq_len * slm->d_model;
    int output_proj_size = slm->d_model * slm->vocab_size;
    
    CHECK_CUDA(cudaMemset(slm->d_token_embedding_grad, 0, token_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_position_embedding_grad, 0, pos_emb_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_output_projection_grad, 0, output_proj_size * sizeof(float)));
    
    zero_gradients_transformer(slm->transformer);
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_input_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Step 4 (backward): Gradient through output projection
    // ∂L/∂output_projection = transformer_output^T * grad_logits
    CHECK_CUBLASLT(cublasLtMatmul(slm->cublaslt_handle,
                                  slm->matmul_TN_desc,
                                  &alpha,
                                  slm->transformer->mlp_layers[slm->num_layers-1]->d_layer_output, slm->transformer_out_for_grad_layout,
                                  slm->d_grad_logits, slm->grad_logits_for_grad_layout,
                                  &alpha,
                                  slm->d_output_projection_grad, slm->output_proj_layout,
                                  slm->d_output_projection_grad, slm->output_proj_layout,
                                  NULL, NULL, 0, 0));
    
    // ∂L/∂transformer_output = grad_logits * output_projection^T
    CHECK_CUBLASLT(cublasLtMatmul(slm->cublaslt_handle,
                                  slm->matmul_NT_desc,
                                  &alpha,
                                  slm->d_grad_logits, slm->logits_layout,
                                  slm->d_output_projection, slm->output_proj_layout,
                                  &beta,
                                  slm->d_grad_embedded, slm->embedded_layout,
                                  slm->d_grad_embedded, slm->embedded_layout,
                                  NULL, NULL, 0, 0));
    
    // Step 3 (backward): Backward pass through transformer
    backward_pass_transformer(slm->transformer, slm->d_embedded_input, slm->d_grad_embedded);
    
    // Step 2 & 1 (backward): Gradient through embeddings
    dim3 grid_emb(slm->batch_size, slm->seq_len);
    dim3 block_emb(slm->d_model);
    
    // Token embedding gradients
    token_embedding_grad_kernel<<<grid_emb, block_emb>>>(
        slm->d_token_embedding_grad, slm->d_grad_embedded, d_input_tokens,
        slm->batch_size, slm->seq_len, slm->d_model
    );
    
    // Position embedding gradients
    dim3 grid_pos(slm->seq_len);
    dim3 block_pos(slm->d_model);
    position_embedding_grad_kernel<<<grid_pos, block_pos>>>(
        slm->d_position_embedding_grad, slm->d_grad_embedded,
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
    
    // Update position embeddings
    int pos_emb_size = slm->seq_len * slm->d_model;
    int pos_blocks = (pos_emb_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<pos_blocks, block_size>>>(
        slm->d_position_embedding, slm->d_position_embedding_grad, slm->d_position_embedding_m, slm->d_position_embedding_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, pos_emb_size, slm->batch_size
    );
    
    // Update output projection
    int output_proj_size = slm->d_model * slm->vocab_size;
    int output_blocks = (output_proj_size + block_size - 1) / block_size;
    adamw_update_kernel_slm<<<output_blocks, block_size>>>(
        slm->d_output_projection, slm->d_output_projection_grad, slm->d_output_projection_m, slm->d_output_projection_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate, slm->weight_decay,
        alpha_t, output_proj_size, slm->batch_size
    );
    
    // Update transformer weights
    update_weights_transformer(slm->transformer, learning_rate);
}

// Save SLM to binary file
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
    
    int token_emb_size = slm->vocab_size * slm->d_model;
    int pos_emb_size = slm->seq_len * slm->d_model;
    int output_proj_size = slm->d_model * slm->vocab_size;
    
    // Allocate host memory and copy embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_position_embedding = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_projection = (float*)malloc(output_proj_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding, slm->d_token_embedding, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_position_embedding, slm->d_position_embedding, pos_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_projection, slm->d_output_projection, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding, sizeof(float), token_emb_size, file);
    fwrite(h_position_embedding, sizeof(float), pos_emb_size, file);
    fwrite(h_output_projection, sizeof(float), output_proj_size, file);
    
    // Save Adam state
    fwrite(&slm->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    float* h_position_embedding_m = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_position_embedding_v = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_projection_m = (float*)malloc(output_proj_size * sizeof(float));
    float* h_output_projection_v = (float*)malloc(output_proj_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_token_embedding_m, slm->d_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_token_embedding_v, slm->d_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_position_embedding_m, slm->d_position_embedding_m, pos_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_position_embedding_v, slm->d_position_embedding_v, pos_emb_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_projection_m, slm->d_output_projection_m, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_projection_v, slm->d_output_projection_v, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(h_token_embedding_v, sizeof(float), token_emb_size, file);
    fwrite(h_position_embedding_m, sizeof(float), pos_emb_size, file);
    fwrite(h_position_embedding_v, sizeof(float), pos_emb_size, file);
    fwrite(h_output_projection_m, sizeof(float), output_proj_size, file);
    fwrite(h_output_projection_v, sizeof(float), output_proj_size, file);
    
    fclose(file);
    
    // Save transformer components
    char transformer_filename[256];
    char base_filename[256];
    
    // Remove .bin extension from filename to create base name
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    save_transformer(slm->transformer, transformer_filename);
    
    // Free temporary host memory
    free(h_token_embedding); free(h_position_embedding); free(h_output_projection);
    free(h_token_embedding_m); free(h_token_embedding_v);
    free(h_position_embedding_m); free(h_position_embedding_v);
    free(h_output_projection_m); free(h_output_projection_v);

    printf("Model saved to %s\n", filename);
}

// Load SLM from binary file
SLM* load_slm(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers, vocab_size;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize SLM
    SLM* slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    int token_emb_size = vocab_size * d_model;
    int pos_emb_size = seq_len * d_model;
    int output_proj_size = d_model * vocab_size;
    
    // Load embeddings
    float* h_token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    float* h_position_embedding = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_projection = (float*)malloc(output_proj_size * sizeof(float));
    
    fread(h_token_embedding, sizeof(float), token_emb_size, file);
    fread(h_position_embedding, sizeof(float), pos_emb_size, file);
    fread(h_output_projection, sizeof(float), output_proj_size, file);
    
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding, h_token_embedding, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_position_embedding, h_position_embedding, pos_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_projection, h_output_projection, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&slm->t, sizeof(int), 1, file);
    
    float* h_token_embedding_m = (float*)malloc(token_emb_size * sizeof(float));
    float* h_token_embedding_v = (float*)malloc(token_emb_size * sizeof(float));
    float* h_position_embedding_m = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_position_embedding_v = (float*)malloc(pos_emb_size * sizeof(float));
    float* h_output_projection_m = (float*)malloc(output_proj_size * sizeof(float));
    float* h_output_projection_v = (float*)malloc(output_proj_size * sizeof(float));
    
    fread(h_token_embedding_m, sizeof(float), token_emb_size, file);
    fread(h_token_embedding_v, sizeof(float), token_emb_size, file);
    fread(h_position_embedding_m, sizeof(float), pos_emb_size, file);
    fread(h_position_embedding_v, sizeof(float), pos_emb_size, file);
    fread(h_output_projection_m, sizeof(float), output_proj_size, file);
    fread(h_output_projection_v, sizeof(float), output_proj_size, file);
    
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding_m, h_token_embedding_m, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_token_embedding_v, h_token_embedding_v, token_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_position_embedding_m, h_position_embedding_m, pos_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_position_embedding_v, h_position_embedding_v, pos_emb_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_projection_m, h_output_projection_m, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_output_projection_v, h_output_projection_v, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    fclose(file);
    
    // Load transformer components
    char transformer_filename[256];
    char base_filename[256];
    
    // Remove .bin extension from filename to create base name  
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(transformer_filename, sizeof(transformer_filename), "%s_transformer.bin", base_filename);
    
    // Free the initialized transformer
    free_transformer(slm->transformer);
    
    // Load the saved transformer
    slm->transformer = load_transformer(transformer_filename, batch_size, cublaslt_handle);
    
    // Free temporary host memory
    free(h_token_embedding); free(h_position_embedding); free(h_output_projection);
    free(h_token_embedding_m); free(h_token_embedding_v);
    free(h_position_embedding_m); free(h_position_embedding_v);
    free(h_output_projection_m); free(h_output_projection_v);
    
    printf("Model loaded from %s\n", filename);
    return slm;
}