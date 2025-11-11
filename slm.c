#include "slm.h"

// Initialize the SLM
SLM* init_slm(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size) {
    SLM* slm = (SLM*)malloc(sizeof(SLM));
    
    // Store dimensions
    slm->seq_len = seq_len;
    slm->d_model = d_model;
    slm->batch_size = batch_size;
    slm->hidden_dim = hidden_dim;
    slm->num_layers = num_layers;
    slm->vocab_size = 65536;
    
    // Initialize Adam parameters
    slm->beta1 = 0.9f;
    slm->beta2 = 0.999f;
    slm->epsilon = 1e-8f;
    slm->t = 0;
    slm->weight_decay = 0.01f;
    
    int token_emb_size = slm->vocab_size * d_model;
    int output_weight_size = d_model * slm->vocab_size;
    int embedded_size = batch_size * seq_len * d_model;
    int output_size = batch_size * seq_len * slm->vocab_size;
    
    // Allocate memory for embeddings and gradients
    slm->token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    slm->token_embedding_grad = (float*)malloc(token_emb_size * sizeof(float));
    
    // Allocate memory for output weights and gradients
    slm->W_output = (float*)malloc(output_weight_size * sizeof(float));
    slm->W_output_grad = (float*)malloc(output_weight_size * sizeof(float));
    
    // Allocate memory for Adam parameters
    slm->token_embedding_m = (float*)calloc(token_emb_size, sizeof(float));
    slm->token_embedding_v = (float*)calloc(token_emb_size, sizeof(float));
    slm->W_output_m = (float*)calloc(output_weight_size, sizeof(float));
    slm->W_output_v = (float*)calloc(output_weight_size, sizeof(float));
    
    // Allocate memory for forward pass buffers
    slm->embedded_input = (float*)malloc(embedded_size * sizeof(float));
    slm->output = (float*)malloc(output_size * sizeof(float));
    
    // Allocate memory for backward pass buffers
    slm->grad_output = (float*)malloc(output_size * sizeof(float));
    
    // Initialize token embeddings
    float token_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < token_emb_size; i++) {
        slm->token_embedding[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Initialize output weights
    for (int i = 0; i < output_weight_size; i++) {
        slm->W_output[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * token_scale;
    }
    
    // Initialize transformer
    slm->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, true);
    
    return slm;
}

// Free SLM memory
void free_slm(SLM* slm) {
    // Free transformer
    free_transformer(slm->transformer);
    
    // Free memory
    free(slm->token_embedding); free(slm->token_embedding_grad);
    free(slm->token_embedding_m); free(slm->token_embedding_v);
    free(slm->W_output); free(slm->W_output_grad);
    free(slm->W_output_m); free(slm->W_output_v);
    free(slm->embedded_input);
    free(slm->output); free(slm->grad_output);
    
    free(slm);
}

// Token embedding lookup
static void token_embedding_lookup(float* embedded, float* token_embedding, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token_idx = b * seq_len + t;
            int token = tokens[token_idx];
            int emb_offset = b * seq_len * d_model + t * d_model;
            int token_emb_offset = token * d_model;
            
            for (int d = 0; d < d_model; d++) {
                embedded[emb_offset + d] = token_embedding[token_emb_offset + d];
            }
        }
    }
}

// Softmax and cross-entropy loss computation
static float softmax_cross_entropy(float* grad_logits, float* logits, unsigned short* targets, int batch_size, int seq_len, int vocab_size) {
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
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
            total_loss += -logf(target_prob + 1e-10f);
            
            // Set gradient: softmax - one_hot
            grad_logits_bt[target_token] -= 1.0f;
        }
    }
    
    return total_loss;
}

// Token embedding gradient accumulation
static void token_embedding_grad_accumulation(float* token_embedding_grad, float* grad_embedded, unsigned short* tokens, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token_idx = b * seq_len + t;
            int token = tokens[token_idx];
            int emb_offset = b * seq_len * d_model + t * d_model;
            int token_emb_offset = token * d_model;
            
            for (int d = 0; d < d_model; d++) {
                token_embedding_grad[token_emb_offset + d] += grad_embedded[emb_offset + d];
            }
        }
    }
}

// Forward pass
void forward_pass_slm(SLM* slm, unsigned short* input_tokens) {
    // Step 1: Token embedding lookup
    token_embedding_lookup(slm->embedded_input, slm->token_embedding, input_tokens,
                          slm->batch_size, slm->seq_len, slm->d_model);
    
    // Step 2: Forward pass through transformer
    forward_pass_transformer(slm->transformer, slm->embedded_input);
    
    // Step 3: Output projection: output = transformer_output * W_output
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                slm->batch_size * slm->seq_len, slm->vocab_size, slm->d_model,
                1.0f, slm->transformer->mlp_layers[slm->num_layers-1]->output, slm->d_model,
                slm->W_output, slm->vocab_size,
                0.0f, slm->output, slm->vocab_size);
}

// Calculate loss
float calculate_loss_slm(SLM* slm, unsigned short* target_tokens) {
    // Compute softmax and cross-entropy loss
    float total_loss = softmax_cross_entropy(slm->grad_output, slm->output, 
                                            target_tokens, slm->batch_size, slm->seq_len, slm->vocab_size);
    
    return total_loss / (slm->batch_size * slm->seq_len);
}

// Zero gradients
void zero_gradients_slm(SLM* slm) {
    int token_emb_size = slm->vocab_size * slm->d_model;
    int output_weight_size = slm->d_model * slm->vocab_size;
    
    memset(slm->token_embedding_grad, 0, token_emb_size * sizeof(float));
    memset(slm->W_output_grad, 0, output_weight_size * sizeof(float));
    
    zero_gradients_transformer(slm->transformer);
}

// Backward pass
void backward_pass_slm(SLM* slm, unsigned short* input_tokens) {
    // Step 3 (backward): Backward pass through output projection
    // grad_W_output = transformer_output^T * grad_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                slm->d_model, slm->vocab_size, slm->batch_size * slm->seq_len,
                1.0f, slm->transformer->mlp_layers[slm->num_layers-1]->output, slm->d_model,
                slm->grad_output, slm->vocab_size,
                1.0f, slm->W_output_grad, slm->vocab_size);
    
    // grad_transformer_output = grad_output * W_output^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                slm->batch_size * slm->seq_len, slm->d_model, slm->vocab_size,
                1.0f, slm->grad_output, slm->vocab_size,
                slm->W_output, slm->vocab_size,
                0.0f, slm->transformer->mlp_layers[slm->num_layers-1]->grad_output, slm->d_model);
    
    // Step 2 (backward): Backward pass through transformer
    backward_pass_transformer(slm->transformer, slm->embedded_input, slm->embedded_input);
    
    // Step 1 (backward): Token embedding gradients
    token_embedding_grad_accumulation(slm->token_embedding_grad, slm->embedded_input, input_tokens,
                                     slm->batch_size, slm->seq_len, slm->d_model);
}

// Update weights
void update_weights_slm(SLM* slm, float learning_rate, int effective_batch_size) {
    slm->t++;
    
    float beta1_t = powf(slm->beta1, slm->t);
    float beta2_t = powf(slm->beta2, slm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int token_emb_size = slm->vocab_size * slm->d_model;
    int output_weight_size = slm->d_model * slm->vocab_size;
    
    // Update token embeddings
    for (int i = 0; i < token_emb_size; i++) {
        float grad = slm->token_embedding_grad[i] / effective_batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        slm->token_embedding_m[i] = slm->beta1 * slm->token_embedding_m[i] + (1.0f - slm->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        slm->token_embedding_v[i] = slm->beta2 * slm->token_embedding_v[i] + (1.0f - slm->beta2) * grad * grad;
        
        float update = alpha_t * slm->token_embedding_m[i] / (sqrtf(slm->token_embedding_v[i]) + slm->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        slm->token_embedding[i] = slm->token_embedding[i] * (1.0f - learning_rate * slm->weight_decay) - update;
    }
    
    // Update output weights
    for (int i = 0; i < output_weight_size; i++) {
        float grad = slm->W_output_grad[i] / effective_batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        slm->W_output_m[i] = slm->beta1 * slm->W_output_m[i] + (1.0f - slm->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        slm->W_output_v[i] = slm->beta2 * slm->W_output_v[i] + (1.0f - slm->beta2) * grad * grad;
        
        float update = alpha_t * slm->W_output_m[i] / (sqrtf(slm->W_output_v[i]) + slm->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        slm->W_output[i] = slm->W_output[i] * (1.0f - learning_rate * slm->weight_decay) - update;
    }
    
    // Update transformer weights
    update_weights_transformer(slm->transformer, learning_rate, effective_batch_size);
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
    int output_weight_size = slm->d_model * slm->vocab_size;
    
    // Save embeddings and weights
    fwrite(slm->token_embedding, sizeof(float), token_emb_size, file);
    fwrite(slm->W_output, sizeof(float), output_weight_size, file);
    
    // Save Adam state
    fwrite(&slm->t, sizeof(int), 1, file);
    fwrite(slm->token_embedding_m, sizeof(float), token_emb_size, file);
    fwrite(slm->token_embedding_v, sizeof(float), token_emb_size, file);
    fwrite(slm->W_output_m, sizeof(float), output_weight_size, file);
    fwrite(slm->W_output_v, sizeof(float), output_weight_size, file);
    
    fclose(file);
    
    // Save transformer
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
    
    printf("Model saved to %s\n", filename);
}

// Load SLM from binary file
SLM* load_slm(const char* filename, int custom_batch_size) {
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
    SLM* slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size);
    
    int token_emb_size = vocab_size * d_model;
    int output_weight_size = d_model * vocab_size;
    
    // Load embeddings and weights
    fread(slm->token_embedding, sizeof(float), token_emb_size, file);
    fread(slm->W_output, sizeof(float), output_weight_size, file);
    
    // Load Adam state
    fread(&slm->t, sizeof(int), 1, file);
    fread(slm->token_embedding_m, sizeof(float), token_emb_size, file);
    fread(slm->token_embedding_v, sizeof(float), token_emb_size, file);
    fread(slm->W_output_m, sizeof(float), output_weight_size, file);
    fread(slm->W_output_v, sizeof(float), output_weight_size, file);
    
    fclose(file);
    
    // Load transformer
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
    slm->transformer = load_transformer(transformer_filename, batch_size);
    
    printf("Model loaded from %s\n", filename);
    return slm;
}