#include "slm.h"

// Initialize SLM
SLM* init_slm(int embed_dim, int seq_len, int num_layers, int mlp_hidden, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->vocab_size = 256;
    slm->embed_dim = embed_dim;
    slm->seq_len = seq_len;
    slm->batch_size = batch_size;
    
    // Initialize AdamW parameters for embeddings
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;

    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&slm->cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(slm->cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Initialize transformer
    slm->transformer = init_transformer(embed_dim, seq_len, batch_size, mlp_hidden, num_layers, slm->cublas_handle);

    // Allocate host memory for embeddings
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));

    // Initialize embeddings on host
    float scale_embeddings = 1.0f / sqrtf(slm->embed_dim);

    for (int i = 0; i < slm->vocab_size * slm->embed_dim; i++) {
        h_embeddings[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_embeddings;
    }

    // Allocate device memory for embeddings
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_grad, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Allocate device memory for working buffers
    CHECK_CUDA(cudaMalloc(&slm->d_embedded_input, batch_size * seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_softmax, batch_size * seq_len * slm->vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_input_gradients, batch_size * seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&slm->d_losses, batch_size * seq_len * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_m, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(slm->d_embeddings_v, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
    
    // Free local host memory
    free(h_embeddings);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    if (slm) {
        // Free transformer
        if (slm->transformer) free_transformer(slm->transformer);
        
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

// CUDA kernel for embedding lookup
__global__ void embedding_lookup_kernel(float* output, float* embeddings, unsigned char* chars, 
                                       int batch_size, int seq_len, int embed_dim) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int embed_idx = threadIdx.x;
    
    if (batch_idx < batch_size && seq_idx < seq_len && embed_idx < embed_dim) {
        int char_idx = chars[batch_idx * seq_len + seq_idx];
        int output_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + embed_idx;
        output[output_idx] = embeddings[char_idx * embed_dim + embed_idx];
    }
}

// CUDA kernel for softmax
__global__ void softmax_kernel(float* output, float* input, int batch_size, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (idx >= total_elements) return;
    
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

// CUDA kernel for cross-entropy loss
__global__ void cross_entropy_loss_kernel(float* losses, float* grad, float* softmax, unsigned char* targets, 
                                         int batch_size, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (idx >= total_elements) return;
    
    float* grad_ptr = grad + idx * vocab_size;
    float* softmax_ptr = softmax + idx * vocab_size;
    int target = targets[idx];
    
    // Compute loss: -log(softmax[target])
    float prob = fmaxf(softmax_ptr[target], 1e-15f); // Avoid log(0)
    losses[idx] = -logf(prob);
    
    // Compute gradient: softmax - one_hot(target)
    for (int i = 0; i < vocab_size; i++) {
        grad_ptr[i] = softmax_ptr[i];
    }
    grad_ptr[target] -= 1.0f;
}

// CUDA kernel for embedding gradient accumulation
__global__ void embedding_gradient_kernel(float* embed_grad, float* input_grad, unsigned char* chars,
                                         int batch_size, int seq_len, int embed_dim) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int embed_idx = threadIdx.x;
    
    if (batch_idx < batch_size && seq_idx < seq_len && embed_idx < embed_dim) {
        int char_idx = chars[batch_idx * seq_len + seq_idx];
        int input_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + embed_idx;
        atomicAdd(&embed_grad[char_idx * embed_dim + embed_idx], input_grad[input_idx]);
    }
}

// CUDA kernel for AdamW update for embeddings
__global__ void adamw_update_kernel_embeddings(float* weight, float* grad, float* m, float* v,
                                              float beta1, float beta2, float epsilon, float learning_rate,
                                              float weight_decay, float alpha_t, int size, int total_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / total_samples;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned char* d_X) {
    // E = W_E[X] - Character embedding lookup
    dim3 block(slm->embed_dim);
    dim3 grid(slm->batch_size, slm->seq_len);
    embedding_lookup_kernel<<<grid, block>>>(
        slm->d_embedded_input, slm->d_embeddings, d_X, 
        slm->batch_size, slm->seq_len, slm->embed_dim
    );
    
    // Forward through transformer
    forward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // Get final transformer output (should be batch_size * seq_len * vocab_size)
    MLP* final_mlp = slm->transformer->mlp_layers[slm->transformer->num_layers - 1];
    
    // P = softmax(logits)
    int total_elements = slm->batch_size * slm->seq_len;
    int blocks = (total_elements + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(
        slm->d_softmax,
        final_mlp->d_layer_output,
        slm->batch_size,
        slm->seq_len,
        slm->vocab_size
    );
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned char* d_y) {
    float loss = 0.0f;
    int total_tokens = slm->batch_size * slm->seq_len;
    
    // Get final transformer output for gradient computation
    MLP* final_mlp = slm->transformer->mlp_layers[slm->transformer->num_layers - 1];
    
    // Compute both cross-entropy loss and logits gradient
    int blocks = (total_tokens + 255) / 256;
    cross_entropy_loss_kernel<<<blocks, 256>>>(
        slm->d_losses, 
        final_mlp->d_error_output, 
        slm->d_softmax, 
        d_y, 
        slm->batch_size,
        slm->seq_len,
        slm->vocab_size
    );
    CHECK_CUBLAS(cublasSasum(slm->cublas_handle, total_tokens, slm->d_losses, 1, &loss));
    
    return loss / total_tokens;
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    zero_gradients_transformer(slm->transformer);
    CHECK_CUDA(cudaMemset(slm->d_embeddings_grad, 0, slm->vocab_size * slm->embed_dim * sizeof(float)));
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned char* d_X) {
    // Backward through transformer (this computes gradients w.r.t. transformer input)
    backward_pass_transformer(slm->transformer, slm->d_embedded_input);
    
    // Get input gradients from first layer of transformer
    Attention* first_attn = slm->transformer->attention_layers[0];
    
    // Copy gradients to input gradients buffer
    CHECK_CUDA(cudaMemcpy(slm->d_input_gradients, first_attn->d_error_output, 
                         slm->batch_size * slm->seq_len * slm->embed_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Accumulate embedding gradients
    dim3 block(slm->embed_dim);
    dim3 grid(slm->batch_size, slm->seq_len);
    embedding_gradient_kernel<<<grid, block>>>(
        slm->d_embeddings_grad, slm->d_input_gradients, d_X, 
        slm->batch_size, slm->seq_len, slm->embed_dim
    );
}

// Update weights
void update_weights_slm(SLM* slm, float learning_rate) {
    // Update transformer weights
    update_weights_transformer(slm->transformer, learning_rate);
    
    // Update embeddings with AdamW
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int embed_size = slm->vocab_size * slm->embed_dim;
    int blocks = (embed_size + 255) / 256;
    int total_samples = slm->batch_size * slm->seq_len;
    
    adamw_update_kernel_embeddings<<<blocks, 256>>>(
        slm->d_embeddings, slm->d_embeddings_grad,
        slm->d_embeddings_m, slm->d_embeddings_v,
        slm->beta1, slm->beta2, slm->epsilon, learning_rate,
        slm->weight_decay, alpha_t, embed_size, total_samples
    );
}

// Save model to binary file
void save_slm(SLM* slm, const char* filename) {
    // Save transformer
    save_transformer(slm->transformer, filename);
    
    // Save embeddings and metadata
    char embed_file[256];
    strcpy(embed_file, filename);
    char* dot = strrchr(embed_file, '.');
    if (dot) *dot = '\0';
    strcat(embed_file, "_embeddings.bin");
    
    // Allocate temporary host memory for embeddings
    float* h_embeddings = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    float* h_embeddings_m = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    float* h_embeddings_v = (float*)malloc(slm->vocab_size * slm->embed_dim * sizeof(float));
    
    // Copy embeddings from device to host
    CHECK_CUDA(cudaMemcpy(h_embeddings, slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_m, slm->d_embeddings_m, slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_embeddings_v, slm->d_embeddings_v, slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
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
    fwrite(&slm->seq_len, sizeof(int), 1, file);
    fwrite(&slm->batch_size, sizeof(int), 1, file);
    fwrite(&slm->t, sizeof(int), 1, file);
    
    // Save embeddings and AdamW state
    fwrite(h_embeddings, sizeof(float), slm->vocab_size * slm->embed_dim, file);
    fwrite(h_embeddings_m, sizeof(float), slm->vocab_size * slm->embed_dim, file);
    fwrite(h_embeddings_v, sizeof(float), slm->vocab_size * slm->embed_dim, file);
    
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
    int vocab_size, embed_dim, seq_len, stored_batch_size, t;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&t, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Allocate temporary host memory for embeddings
    float* h_embeddings = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_embeddings_m = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* h_embeddings_v = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    
    // Read embeddings and AdamW state
    fread(h_embeddings, sizeof(float), vocab_size * embed_dim, file);
    fread(h_embeddings_m, sizeof(float), vocab_size * embed_dim, file);
    fread(h_embeddings_v, sizeof(float), vocab_size * embed_dim, file);
    fclose(file);
    
    // Load transformer
    Transformer* loaded_transformer = load_transformer(filename, batch_size, NULL);
    if (!loaded_transformer) {
        free(h_embeddings);
        free(h_embeddings_m);
        free(h_embeddings_v);
        return NULL;
    }
    
    // Initialize SLM with loaded parameters
    int mlp_hidden = loaded_transformer->mlp_hidden;
    int num_layers = loaded_transformer->num_layers;
    
    SLM* slm = init_slm(embed_dim, seq_len, num_layers, mlp_hidden, batch_size);
    
    // Replace the initialized transformer with loaded one
    free_transformer(slm->transformer);
    slm->transformer = loaded_transformer;
    slm->transformer->cublas_handle = slm->cublas_handle;
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings, h_embeddings, vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_m, h_embeddings_m, vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(slm->d_embeddings_v, h_embeddings_v, vocab_size * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    slm->t = t;
    
    free(h_embeddings);
    free(h_embeddings_m);
    free(h_embeddings_v);
    
    printf("Model loaded from %s\n", filename);
    return slm;
}

// Text generation function
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature, float top_p) {
    int seed_len = strlen(seed_text);
    if (seed_len == 0) {
        printf("Error: Empty seed text\n");
        return;
    }
    
    int total_len = seed_len + generation_length;
    if (total_len > slm->seq_len) {
        printf("Error: Total generation length (%d) exceeds model seq_len (%d)\n", total_len, slm->seq_len);
        return;
    }
    
    // Create a temporary SLM instance for generation with batch_size=1
    SLM* gen_slm = init_slm(slm->embed_dim, slm->seq_len, slm->transformer->num_layers, 
                           slm->transformer->mlp_hidden, 1);
    
    // Copy trained weights from main model to generation model
    // Copy transformer weights
    for (int i = 0; i < slm->transformer->num_layers; i++) {
        Attention* src_attn = slm->transformer->attention_layers[i];
        Attention* dst_attn = gen_slm->transformer->attention_layers[i];
        MLP* src_mlp = slm->transformer->mlp_layers[i];
        MLP* dst_mlp = gen_slm->transformer->mlp_layers[i];
        
        // Copy attention weights
        int attn_weight_size = slm->embed_dim * slm->embed_dim;
        CHECK_CUDA(cudaMemcpy(dst_attn->d_W_q, src_attn->d_W_q, attn_weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_attn->d_W_k, src_attn->d_W_k, attn_weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_attn->d_W_v, src_attn->d_W_v, attn_weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_attn->d_W_o, src_attn->d_W_o, attn_weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Copy MLP weights
        int w1_size = slm->transformer->mlp_hidden * slm->embed_dim;
        int w2_size = (i == slm->transformer->num_layers - 1) ? slm->vocab_size * slm->transformer->mlp_hidden : slm->embed_dim * slm->transformer->mlp_hidden;
        CHECK_CUDA(cudaMemcpy(dst_mlp->d_W1, src_mlp->d_W1, w1_size * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dst_mlp->d_W2, src_mlp->d_W2, w2_size * sizeof(float), cudaMemcpyDeviceToDevice));
    } 
    
    // Copy embeddings
    CHECK_CUDA(cudaMemcpy(gen_slm->d_embeddings, slm->d_embeddings, slm->vocab_size * slm->embed_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Allocate host memory for sequence and probabilities
    unsigned char* h_sequence = (unsigned char*)malloc(slm->seq_len * sizeof(unsigned char));
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    int* indices = (int*)malloc(slm->vocab_size * sizeof(int));
    unsigned char* d_sequence;
    CHECK_CUDA(cudaMalloc(&d_sequence, slm->seq_len * sizeof(unsigned char)));
    
    // Initialize sequence with seed
    memset(h_sequence, 0, slm->seq_len);
    for (int i = 0; i < seed_len && i < slm->seq_len; i++) {
        h_sequence[i] = (unsigned char)seed_text[i];
    }
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    for (int gen_step = 0; gen_step < generation_length; gen_step++) {
        int current_pos = seed_len + gen_step;
        if (current_pos >= slm->seq_len) break;
        
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_sequence, h_sequence, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_slm(gen_slm, d_sequence);
        
        // Get probabilities for the current position
        float* d_probs = gen_slm->d_softmax + (current_pos - 1) * gen_slm->vocab_size;
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_probs, d_probs, gen_slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
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
        
        // Add to sequence
        h_sequence[current_pos] = (unsigned char)sampled_char;
    }
    
    printf("\n");
    
    // Cleanup
    free(h_sequence);
    free(h_probs);
    free(indices);
    cudaFree(d_sequence);
    free_slm(gen_slm);
}