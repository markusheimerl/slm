#include "slm.h"

// Initialize SLM
SLM* init_slm(int embed_dim, int state_dim, int seq_len, int num_layers, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    slm->num_layers = num_layers;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(slm->cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Initialize SSM layers
    slm->ssms = (SSM**)malloc(num_layers * sizeof(SSM*));
    
    for (int i = 0; i < num_layers; i++) {
        int output_dim = (i == num_layers - 1) ? slm->vocab_size : embed_dim;
        slm->ssms[i] = init_ssm(embed_dim, state_dim, output_dim, seq_len, batch_size, slm->cublas_handle);
    }

    // Allocate host memory for embeddings
    __half* h_embeddings = (__half*)malloc(slm->vocab_size * slm->embed_dim * sizeof(__half));

    // Initialize embeddings on host
    float scale_embeddings = 1.0f / sqrtf(slm->vocab_size);

    for (int i = 0; i < slm->vocab_size * slm->embed_dim; i++) {
        float val = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_embeddings;
        h_embeddings[i] = __float2half(val);
    }

    // Allocate device memory for embeddings
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    
    // Allocate device memory for working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, seq_len * batch_size * embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, seq_len * batch_size * slm->vocab_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, seq_len * batch_size * embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, seq_len * batch_size * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, slm->vocab_size * slm->embed_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_m, 0, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_v, 0, slm->vocab_size * slm->embed_dim * sizeof(__half)));
    
    // Free local host memory
    free(h_embeddings);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    if (slm) {
        // Free SSM layers
        for (int i = 0; i < slm->num_layers; i++) {
            if (slm->ssms[i]) free_ssm(slm->ssms[i]);
        }
        free(slm->ssms);
        
        // Free device memory
        cudaFree(slm->d_embeddings);
        cudaFree(slm->d_embeddings_grad);
        cudaFree(slm->d_embeddings_m);
        cudaFree(slm->d_embeddings_v);
        cudaFree(slm->d_embedded_input);
        cudaFree(slm->d_softmax);
        cudaFree(slm->d_input_gradients);
        cudaFree(slm->d_losses);
        
        CHECK_CUBLAS(cublasDestroy(slm->cublas_handle));
        free(slm);
    }
}

// Reset state for new sequence
void reset_state_slm(SLM* slm) {
    for (int i = 0; i < slm->num_layers; i++) {
        reset_state_ssm(slm->ssms[i]);
    }
}

// CUDA kernel for embedding lookup (FP16)
__global__ void embedding_lookup_kernel_fp16(__half* output, __half* embeddings, unsigned char* chars, 
                                             int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int char_idx = chars[idx];
        for (int i = 0; i < embed_dim; i++) {
            output[idx * embed_dim + i] = embeddings[char_idx * embed_dim + i];
        }
    }
}

// CUDA kernel for softmax (FP16)
__global__ void softmax_kernel_fp16(__half* output, __half* input, int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    __half* in = input + idx * vocab_size;
    __half* out = output + idx * vocab_size;
    
    // Find max for numerical stability
    __half max_val = in[0];
    for (int i = 1; i < vocab_size; i++) {
        max_val = __hmax(max_val, in[i]);
    }
    
    // Compute exp and sum
    __half sum = __float2half(0.0f);
    for (int i = 0; i < vocab_size; i++) {
        out[i] = hexp(__hsub(in[i], max_val));
        sum = __hadd(sum, out[i]);
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        out[i] = __hdiv(out[i], sum);
    }
}

// CUDA kernel for cross-entropy loss (FP16 input, FP32 loss)
__global__ void cross_entropy_loss_kernel_fp16(float* losses, __half* grad, __half* softmax, unsigned char* targets, 
                                               int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    __half* grad_ptr = grad + idx * vocab_size;
    __half* softmax_ptr = softmax + idx * vocab_size;
    int target = targets[idx];
    
    // Compute loss: -log(softmax[target])
    __half prob = __hmax(softmax_ptr[target], __float2half(1e-7f)); // Avoid log(0)
    losses[idx] = -logf(__half2float(prob));
    
    // Compute gradient: softmax - one_hot(target)
    for (int i = 0; i < vocab_size; i++) {
        grad_ptr[i] = softmax_ptr[i];
    }
    grad_ptr[target] = __hsub(grad_ptr[target], __float2half(1.0f));
}

// CUDA kernel for embedding gradient accumulation (FP16)
__global__ void embedding_gradient_kernel_fp16(__half* embed_grad, __half* input_grad, unsigned char* chars,
                                               int batch_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        int char_idx = chars[idx];
        for (int i = 0; i < embed_dim; i++) {
            // Atomic add for FP16
            atomicAdd(&embed_grad[char_idx * embed_dim + i], input_grad[idx * embed_dim + i]);
        }
    }
}

// CUDA kernel for AdamW update (FP16)
__global__ void adamw_update_kernel_slm(__half* weight, __half* grad, __half* m, __half* v,
                                        __half beta1, __half beta2, __half epsilon, __half learning_rate,
                                        __half weight_decay, __half alpha_t, int size, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __half g = __hdiv(grad[idx], __float2half((float)total_samples));
        __half one = __float2half(1.0f);
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        __half one_minus_beta1 = __hsub(one, beta1);
        m[idx] = __hadd(__hmul(beta1, m[idx]), __hmul(one_minus_beta1, g));
        
        // v = β₂v + (1-β₂)(∂L/∂W)²
        __half one_minus_beta2 = __hsub(one, beta2);
        __half g_squared = __hmul(g, g);
        v[idx] = __hadd(__hmul(beta2, v[idx]), __hmul(one_minus_beta2, g_squared));
        
        __half update = __hdiv(__hmul(alpha_t, m[idx]), __hadd(hsqrt(v[idx]), epsilon));
        
        // W = (1-λη)W - update
        __half decay_factor = __hsub(one, __hmul(learning_rate, weight_decay));
        weight[idx] = __hsub(__hmul(weight[idx], decay_factor), update);
    }
}

// Forward pass for single timestep
void forward_pass_slm(SLM* slm, unsigned char* d_X_t, int timestep) {
    int batch_size = slm->ssms[0]->batch_size;
    
    // E_t = W_E[X_t] - Character embedding lookup
    __half* d_embedded_t = slm->d_embedded_input + timestep * batch_size * slm->embed_dim;
    
    dim3 block(256);
    dim3 grid((batch_size + 255) / 256);
    embedding_lookup_kernel_fp16<<<grid, block>>>(d_embedded_t, slm->d_embeddings, d_X_t, batch_size, slm->embed_dim);
    
    __half* current_input = d_embedded_t;
    
    // Forward through all SSM layers
    for (int layer = 0; layer < slm->num_layers; layer++) {
        forward_pass_ssm(slm->ssms[layer], current_input, timestep);
        
        if (layer < slm->num_layers - 1) {
            current_input = slm->ssms[layer]->d_layer2_output + timestep * batch_size * slm->embed_dim;
        }
    }

    // P_t = softmax(logits_t)
    __half* d_softmax_t = slm->d_softmax + timestep * batch_size * slm->vocab_size;
    int blocks = (batch_size + 255) / 256;
    softmax_kernel_fp16<<<blocks, 256>>>(
        d_softmax_t,
        slm->ssms[slm->num_layers - 1]->d_layer2_output + timestep * batch_size * slm->vocab_size,
        batch_size,
        slm->vocab_size
    );
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    float loss = 0.0f;
    int seq_len = slm->ssms[0]->seq_len;
    int batch_size = slm->ssms[0]->batch_size;
    int total_tokens = seq_len * batch_size;
    
    // Compute both cross-entropy loss and logits gradient
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_loss_kernel_fp16<<<blocks, 256>>>(
        slm->d_losses, 
        slm->ssms[slm->num_layers - 1]->d_error_output, 
        slm->d_softmax, 
        d_y, 
        total_tokens, 
        slm->vocab_size
    );
    
    // Sum losses (using cublas for FP32 sum)
    float alpha = 1.0f;
    CHECK_CUBLAS(cublasSasum(slm->cublas_handle, total_tokens, slm->d_losses, 1, &loss));
    
    return loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    for (int i = 0; i < slm->num_layers; i++) {
        zero_gradients_ssm(slm->ssms[i]);
    }
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, slm->vocab_size * slm->embed_dim * sizeof(__half)));
}

// Backward pass for single timestep
void backward_pass_slm(SLM* slm, unsigned char* d_X_t, int timestep) {
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    int batch_size = slm->ssms[0]->batch_size;
    
    for (int layer = slm->num_layers - 1; layer >= 1; layer--) {
        // Input comes from previous layer's output
        __half* layer_input_t = slm->ssms[layer - 1]->d_layer2_output + timestep * batch_size * slm->embed_dim;
        
        // Backward SSM of current layer for this timestep
        backward_pass_ssm(slm->ssms[layer], layer_input_t, timestep);
        
        // Compute input gradients
        __half* input_grad_t = slm->d_input_gradients + timestep * batch_size * slm->embed_dim;
        __half* error_hidden_t = slm->ssms[layer]->d_error_hidden + timestep * batch_size * slm->ssms[layer]->state_dim;
        __half* error_output_t = slm->ssms[layer]->d_error_output + timestep * batch_size * slm->ssms[layer]->output_dim;
        
        // ∂L/∂X = B^T (∂L/∂H) + D^T (∂L/∂Y)
        CHECK_CUBLAS(cublasHgemm(slm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssms[layer]->state_dim,
                                &alpha, slm->ssms[layer]->d_B, slm->embed_dim,
                                error_hidden_t, slm->ssms[layer]->state_dim,
                                &beta, input_grad_t, slm->embed_dim));
        
        CHECK_CUBLAS(cublasHgemm(slm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                slm->embed_dim, batch_size, slm->ssms[layer]->output_dim,
                                &alpha, slm->ssms[layer]->d_D, slm->embed_dim,
                                error_output_t, slm->ssms[layer]->output_dim,
                                &alpha, input_grad_t, slm->embed_dim));
        
        // Copy to previous layer's error buffer
        __half* prev_error_output_t = slm->ssms[layer - 1]->d_error_output + timestep * batch_size * slm->embed_dim;
        CHECK_CUDA(cudaMemcpy(prev_error_output_t, input_grad_t, 
                             batch_size * slm->embed_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
    }
    
    __half* layer_input_t = slm->d_embedded_input + timestep * batch_size * slm->embed_dim;
    backward_pass_ssm(slm->ssms[0], layer_input_t, timestep);
    
    // Compute embedding gradients from first layer
    __half* embed_grad_t = slm->d_input_gradients + timestep * batch_size * slm->embed_dim;
    __half* error_hidden_t = slm->ssms[0]->d_error_hidden + timestep * batch_size * slm->ssms[0]->state_dim;
    __half* error_output_t = slm->ssms[0]->d_error_output + timestep * batch_size * slm->ssms[0]->output_dim;
    
    // ∂L/∂E = B₀^T (∂L/∂H₀) + D₀^T (∂L/∂Y₀)
    CHECK_CUBLAS(cublasHgemm(slm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embed_dim, batch_size, slm->ssms[0]->state_dim,
                            &alpha, slm->ssms[0]->d_B, slm->embed_dim,
                            error_hidden_t, slm->ssms[0]->state_dim,
                            &beta, embed_grad_t, slm->embed_dim));
    
    CHECK_CUBLAS(cublasHgemm(slm->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            slm->embed_dim, batch_size, slm->ssms[0]->output_dim,
                            &alpha, slm->ssms[0]->d_D, slm->embed_dim,
                            error_output_t, slm->ssms[0]->output_dim,
                            &alpha, embed_grad_t, slm->embed_dim));
    
    // Accumulate embedding gradients
    dim3 block(256), grid((batch_size + 255) / 256);
    embedding_gradient_kernel_fp16<<<grid, block>>>(
        slm->d_embeddings_grad, embed_grad_t, d_X_t, batch_size, slm->embed_dim
    );
}

// Update weights using AdamW
void update_weights_slm(SLM* slm, __half learning_rate) {
    // Update all SSM weights
    for (int i = 0; i < slm->num_layers; i++) {
        update_weights_ssm(slm->ssms[i], learning_rate);
    }
    
    // Update embeddings using AdamW
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    
    float beta1_t_f = powf(__half2float(slm->ssms[0]->beta1), slm->ssms[0]->t);
    float beta2_t_f = powf(__half2float(slm->ssms[0]->beta2), slm->ssms[0]->t);
    float alpha_t_f = __half2float(learning_rate) * sqrtf(1.0f - beta2_t_f) / (1.0f - beta1_t_f);
    __half alpha_t = __float2half(alpha_t_f);
    
    adamw_update_kernel_slm<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->ssms[0]->beta1, slm->ssms[0]->beta2, slm->ssms[0]->epsilon,
        learning_rate, slm->ssms[0]->weight_decay, alpha_t,
        embed_size, slm->ssms[0]->seq_len * slm->ssms[0]->batch_size
    );
}

// Save model to binary file
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
    
    // Save embeddings and metadata
    char embed_file[256];
    strcpy(embed_file, filename);
    dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    // Allocate temporary host memory for embeddings
    __half* h_embeddings = (__half*)malloc(slm->vocab_size * slm->embed_dim * sizeof(__half));
    __half* h_embeddings_m = (__half*)malloc(slm->vocab_size * slm->embed_dim * sizeof(__half));
    __half* h_embeddings_v = (__half*)malloc(slm->vocab_size * slm->embed_dim * sizeof(__half));
    
    // Copy embeddings from device to host
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_m, slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_v, slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(__half), cudaMemcpyDeviceToHost));
    
    FILE* file = fopen(embed_file, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", embed_file);
        free(h_embeddings);
        free(h_embeddings_m);
        free(h_embeddings_v);
        return;
    }
    
    // Save dimensions and metadata
    fwrite(&slm->vocab_size, sizeof(int), 1, file);
    fwrite(&slm->embed_dim, sizeof(int), 1, file);
    fwrite(&slm->num_layers, sizeof(int), 1, file);
    fwrite(&slm->ssms[0]->seq_len, sizeof(int), 1, file);
    fwrite(&slm->ssms[0]->state_dim, sizeof(int), 1, file);
    fwrite(&slm->ssms[0]->batch_size, sizeof(int), 1, file);
    
    // Save embeddings and Adam state
    fwrite(h_embeddings, sizeof(__half), slm->vocab_size * slm->embed_dim, file);
    fwrite(h_embeddings_m, sizeof(__half), slm->vocab_size * slm->embed_dim, file);
    fwrite(h_embeddings_v, sizeof(__half), slm->vocab_size * slm->embed_dim, file);
    
    // Free temporary host memory
    free(h_embeddings);
    free(h_embeddings_m);
    free(h_embeddings_v);
    
    fclose(file);
    printf("Embeddings saved to %s\n", embed_file);
}

// Load model from binary file
SLM* load_slm(const char* filename, int custom_batch_size) {
    // Load embeddings and metadata first
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    FILE* file = fopen(embed_file, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", embed_file);
        return NULL;
    }
    
    // Read dimensions and metadata
    int vocab_size, embed_dim, num_layers, seq_len, state_dim, stored_batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Allocate temporary host memory for embeddings
    __half* h_embeddings = (__half*)malloc(vocab_size * embed_dim * sizeof(__half));
    __half* h_embeddings_m = (__half*)malloc(vocab_size * embed_dim * sizeof(__half));
    __half* h_embeddings_v = (__half*)malloc(vocab_size * embed_dim * sizeof(__half));
    
    // Read embeddings and Adam state
    fread(h_embeddings, sizeof(__half), vocab_size * embed_dim, file);
    fread(h_embeddings_m, sizeof(__half), vocab_size * embed_dim, file);
    fread(h_embeddings_v, sizeof(__half), vocab_size * embed_dim, file);
    fclose(file);
    
    // Initialize SLM with loaded parameters
    SLM* slm = init_slm(embed_dim, state_dim, seq_len, num_layers, batch_size);
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, vocab_size * embed_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_m, h_embeddings_m, vocab_size * embed_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_v, h_embeddings_v, vocab_size * embed_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    free(h_embeddings);
    free(h_embeddings_m);
    free(h_embeddings_v);
    
    // Load SSMs
    for (int i = 0; i < num_layers; i++) {
        // Load SSM
        char ssm_file[256];
        strcpy(ssm_file, filename);
        dot = strrchr(ssm_file, '.');
        if (dot) *dot = '\0';
        char suffix[16];
        sprintf(suffix, "_ssm%d.bin", i);
        strcat(ssm_file, suffix);
        
        SSM* loaded_ssm = load_ssm(ssm_file, batch_size, slm->cublas_handle);
        if (loaded_ssm) {
            free_ssm(slm->ssms[i]);
            slm->ssms[i] = loaded_ssm;
        }
    }
    
    printf("Model loaded from %s (with %d layers)\n", filename, num_layers);
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
    SLM* gen_slm = init_slm(slm->embed_dim, slm->ssms[0]->state_dim, seed_len + generation_length, slm->num_layers, 1);
    
    // Copy trained weights from main model to generation model
    for (int i = 0; i < slm->num_layers; i++) {
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_A, slm->ssms[i]->d_A, slm->ssms[i]->state_dim * slm->ssms[i]->state_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_B, slm->ssms[i]->d_B, slm->ssms[i]->state_dim * slm->ssms[i]->input_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_C, slm->ssms[i]->d_C, slm->ssms[i]->output_dim * slm->ssms[i]->state_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(gen_slm->ssms[i]->d_D, slm->ssms[i]->d_D, slm->ssms[i]->output_dim * slm->ssms[i]->input_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
    } 
    
    // Copy embeddings
    CHECK_CUDA(cudaMemcpy(gen_slm->d_embeddings, slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(__half), cudaMemcpyDeviceToDevice));
    
    // Allocate temporary buffers for generation
    unsigned char* h_char = (unsigned char*)malloc(sizeof(unsigned char));
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    int* indices = (int*)malloc(slm->vocab_size * sizeof(int));
    unsigned char* d_char;
    __half* d_probs_fp16 = NULL;
    CHECK_CUDA(cudaMalloc(&d_char, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_probs_fp16, gen_slm->vocab_size * sizeof(__half)));
    
    // Reset state for generation
    reset_state_slm(gen_slm);
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    for (int i = 0; i < seed_len + generation_length; i++) {
        if (i < seed_len) {
            h_char[0] = (unsigned char)seed_text[i];
        }

        CHECK_CUDA(cudaMemcpy(d_char, h_char, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass for this timestep
        forward_pass_slm(gen_slm, d_char, i);
        
        // Only do sampling during generation phase
        if (i >= seed_len) {
            // Get the final logits from the softmax buffer (FP16)
            __half* d_probs = gen_slm->d_softmax + i * gen_slm->vocab_size;
            
            // Copy probabilities to host and convert to FP32
            CHECK_CUDA(cudaMemcpy(d_probs_fp16, d_probs, gen_slm->vocab_size * sizeof(__half), cudaMemcpyDeviceToHost));
            for (int j = 0; j < gen_slm->vocab_size; j++) {
                h_probs[j] = __half2float(d_probs_fp16[j]);
            }
            
            // Apply temperature scaling
            if (temperature != 1.0f) {
                float sum = 0.0f;
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    h_probs[j] = expf(logf(h_probs[j] + 1e-15f) / temperature);
                    sum += h_probs[j];
                }
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    h_probs[j] /= sum;
                }
            }
            
            // Apply nucleus (top-p) sampling
            if (top_p < 1.0f && top_p > 0.0f) {
                // Initialize indices
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    indices[j] = j;
                }
                
                // Sort by probability (bubble sort for simplicity)
                for (int k = 0; k < gen_slm->vocab_size - 1; k++) {
                    for (int j = k + 1; j < gen_slm->vocab_size; j++) {
                        if (h_probs[indices[k]] < h_probs[indices[j]]) {
                            int temp = indices[k];
                            indices[k] = indices[j];
                            indices[j] = temp;
                        }
                    }
                }
                
                // Find cutoff point
                float cumulative_prob = 0.0f;
                int cutoff = 0;
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    cumulative_prob += h_probs[indices[j]];
                    if (cumulative_prob >= top_p) {
                        cutoff = j + 1;
                        break;
                    }
                }
                
                // Zero out probabilities beyond cutoff
                for (int j = cutoff; j < gen_slm->vocab_size; j++) {
                    h_probs[indices[j]] = 0.0f;
                }
                
                // Renormalize
                float sum = 0.0f;
                for (int j = 0; j < gen_slm->vocab_size; j++) {
                    sum += h_probs[j];
                }
                if (sum > 0.0f) {
                    for (int j = 0; j < gen_slm->vocab_size; j++) {
                        h_probs[j] /= sum;
                    }
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
            h_char[0] = (unsigned char)sampled_char;
        }
    }
    
    printf("\n");
    
    // Cleanup
    free(h_char);
    free(h_probs);
    free(indices);
    cudaFree(d_char);
    cudaFree(d_probs_fp16);
    free_slm(gen_slm);
}