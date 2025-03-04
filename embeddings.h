#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------
// Structure for holding embeddings and their gradients
// ---------------------------------------------------------------------
typedef struct {
    // Embeddings
    float* embeddings;       // vocab_size x embedding_dim
    
    // Gradients
    float* embedding_grads;  // vocab_size x embedding_dim
    
    // Adam optimizer first (m) and second (v) moment estimates
    float* embedding_m;      // First moment
    float* embedding_v;      // Second moment
    
    // Adam hyperparameters and counter
    float beta1;             // e.g., 0.9
    float beta2;             // e.g., 0.999
    float epsilon;           // e.g., 1e-8
    float weight_decay;      // e.g., 0.01
    int adam_t;              // time step counter
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
} Embeddings;

// ---------------------------------------------------------------------
// Function: Embed input bytes (forward pass)
// ---------------------------------------------------------------------
void embed_bytes(float* output, 
                const unsigned char* bytes, 
                const float* embeddings, 
                int batch_size, 
                int embedding_dim) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding table
        int embedding_offset = byte_val * embedding_dim;
        
        // Copy the embedding vector to output
        memcpy(&output[batch_idx * embedding_dim], 
               &embeddings[embedding_offset], 
               embedding_dim * sizeof(float));
    }
}

// ---------------------------------------------------------------------
// Function: Compute embedding gradients (backward pass)
// ---------------------------------------------------------------------
void embedding_gradient(float* embedding_grads,
                       const float* input_grads,
                       const unsigned char* bytes,
                       int batch_size,
                       int embedding_dim) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding gradient table
        int grad_offset = byte_val * embedding_dim;
        
        // Add gradient to embedding gradient table
        for (int emb_idx = 0; emb_idx < embedding_dim; emb_idx++) {
            embedding_grads[grad_offset + emb_idx] += 
                input_grads[batch_idx * embedding_dim + emb_idx];
        }
    }
}

// ---------------------------------------------------------------------
// Function: AdamW update for embeddings
// ---------------------------------------------------------------------
void adamw_embeddings(float* W, const float* grad, float* m, float* v, 
                     int size, float beta1, float beta2, float epsilon, 
                     float weight_decay, float learning_rate, int batch_size, 
                     float bias_correction1, float bias_correction2) {
    for (int idx = 0; idx < size; idx++) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// Function: Initialize embeddings
// Initializes the embeddings structure, allocates memory,
// sets initial weights with scaled random values.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
Embeddings* init_embeddings(int vocab_size, int embedding_dim) {
    Embeddings* emb = (Embeddings*)malloc(sizeof(Embeddings));
    emb->vocab_size = vocab_size;
    emb->embedding_dim = embedding_dim;
    
    // Set Adam hyperparameters
    emb->beta1 = 0.9f;
    emb->beta2 = 0.999f;
    emb->epsilon = 1e-8f;
    emb->weight_decay = 0.01f;
    emb->adam_t = 0;
    
    // Allocate memory for embeddings
    emb->embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    
    // Initialize matrices with scaled random values
    float scale = 1.0f / sqrtf(embedding_dim);
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        emb->embeddings[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    
    // Allocate memory for gradients
    emb->embedding_grads = (float*)calloc(vocab_size * embedding_dim, sizeof(float));
    
    // Allocate memory for Adam first and second moment estimates and initialize to zero
    emb->embedding_m = (float*)calloc(vocab_size * embedding_dim, sizeof(float));
    emb->embedding_v = (float*)calloc(vocab_size * embedding_dim, sizeof(float));
    
    return emb;
}

// ---------------------------------------------------------------------
// Function: Forward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_forward(Embeddings* emb, unsigned char* bytes, float* output, int batch_size) {
    embed_bytes(output, bytes, emb->embeddings, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Backward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_backward(Embeddings* emb, float* input_grads, unsigned char* bytes, int batch_size) {
    embedding_gradient(emb->embedding_grads, input_grads, bytes, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Zero gradients for embeddings
// ---------------------------------------------------------------------
void zero_embedding_gradients(Embeddings* emb) {
    size_t emb_size = emb->vocab_size * emb->embedding_dim * sizeof(float);
    memset(emb->embedding_grads, 0, emb_size);
}

// ---------------------------------------------------------------------
// Function: Update embeddings with AdamW optimizer
// ---------------------------------------------------------------------
void update_embeddings(Embeddings* emb, float learning_rate, int batch_size) {
    emb->adam_t++; // Increment time step
    
    float bias_correction1 = 1.0f - powf(emb->beta1, (float)emb->adam_t);
    float bias_correction2 = 1.0f - powf(emb->beta2, (float)emb->adam_t);
    
    int size = emb->vocab_size * emb->embedding_dim;
    
    adamw_embeddings(
        emb->embeddings, emb->embedding_grads, emb->embedding_m, emb->embedding_v,
        size, emb->beta1, emb->beta2, emb->epsilon, emb->weight_decay, 
        learning_rate, batch_size, bias_correction1, bias_correction2);
}

// ---------------------------------------------------------------------
// Function: Free embeddings
// Frees all allocated memory.
// ---------------------------------------------------------------------
void free_embeddings(Embeddings* emb) {
    if (!emb) return;
    
    free(emb->embeddings);
    free(emb->embedding_grads);
    free(emb->embedding_m);
    free(emb->embedding_v);
    free(emb);
}

// ---------------------------------------------------------------------
// Function: Save embeddings
// Saves the embeddings to a binary file.
// ---------------------------------------------------------------------
void save_embeddings(Embeddings* emb, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write dimensions
    fwrite(&emb->vocab_size, sizeof(int), 1, file);
    fwrite(&emb->embedding_dim, sizeof(int), 1, file);
    
    // Write Adam hyperparameters
    fwrite(&emb->beta1, sizeof(float), 1, file);
    fwrite(&emb->beta2, sizeof(float), 1, file);
    fwrite(&emb->epsilon, sizeof(float), 1, file);
    fwrite(&emb->weight_decay, sizeof(float), 1, file);
    fwrite(&emb->adam_t, sizeof(int), 1, file);
    
    // Write data to file
    fwrite(emb->embeddings, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    fwrite(emb->embedding_m, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    fwrite(emb->embedding_v, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    
    fclose(file);
    printf("Embeddings saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: Load embeddings
// Loads the embeddings from a binary file and initializes.
// ---------------------------------------------------------------------
Embeddings* load_embeddings(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int vocab_size, embedding_dim;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embedding_dim, sizeof(int), 1, file);
    
    Embeddings* emb = init_embeddings(vocab_size, embedding_dim);
    
    // Read Adam hyperparameters
    fread(&emb->beta1, sizeof(float), 1, file);
    fread(&emb->beta2, sizeof(float), 1, file);
    fread(&emb->epsilon, sizeof(float), 1, file);
    fread(&emb->weight_decay, sizeof(float), 1, file);
    fread(&emb->adam_t, sizeof(int), 1, file);
    
    // Read data from file
    fread(emb->embeddings, sizeof(float), vocab_size * embedding_dim, file);
    fread(emb->embedding_m, sizeof(float), vocab_size * embedding_dim, file);
    fread(emb->embedding_v, sizeof(float), vocab_size * embedding_dim, file);
    
    fclose(file);
    printf("Embeddings loaded from %s\n", filename);
    return emb;
}

// ---------------------------------------------------------------------
// Function: Softmax for probabilities output
// ---------------------------------------------------------------------
void softmax(float* logits, int batch_size, int vocab_size) {
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Get pointer to this batch item's prediction vector
        float* batch_logits = logits + batch_idx * vocab_size;
        
        // Find max value for numerical stability
        float max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
        
        // Compute exp(logits - max) and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] = expf(batch_logits[i] - max_val);
            sum_exp += batch_logits[i];
        }
        
        // Ensure sum is not zero
        sum_exp = fmaxf(sum_exp, 1e-10f);
        
        // Normalize to get probabilities
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] /= sum_exp;
        }
    }
}

// ---------------------------------------------------------------------
// Function: Cross-entropy loss computation
// ---------------------------------------------------------------------
float cross_entropy_loss(float* error, 
                        const float* probs, 
                        const float* targets, 
                        int batch_size, 
                        int vocab_size) {
    float total_loss = 0.0f;
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const float* batch_probs = probs + batch_idx * vocab_size;
        const float* batch_target = targets + batch_idx * vocab_size;
        float* batch_error = error + batch_idx * vocab_size;
        float batch_loss = 0.0f;
        
        for (int i = 0; i < vocab_size; i++) {
            float prob = fmaxf(fminf(batch_probs[i], 1.0f - 1e-7f), 1e-7f);
            if (batch_target[i] > 0.0f) {
                batch_loss -= batch_target[i] * logf(prob);
            }
            
            // Gradient for softmax with cross-entropy: prob - target
            batch_error[i] = batch_probs[i] - batch_target[i];
        }
        
        total_loss += batch_loss;
    }
    
    return total_loss / batch_size;
}

// ---------------------------------------------------------------------
// Function: Calculate cross-entropy loss
// ---------------------------------------------------------------------
float calculate_cross_entropy_loss(SSM* ssm, float* y) {
    // Apply softmax to get probabilities
    softmax(ssm->predictions, ssm->batch_size, ssm->output_dim);
    
    // Compute cross-entropy loss and gradients
    float loss = cross_entropy_loss(ssm->error, 
                                   ssm->predictions, 
                                   y, 
                                   ssm->batch_size, 
                                   ssm->output_dim);
    
    return loss;
}

#endif // EMBEDDINGS_H