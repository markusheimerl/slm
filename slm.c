#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"

// ---------------------------------------------------------------------
// Structure for holding embeddings and their gradients
// ---------------------------------------------------------------------
typedef struct {
    // Embeddings (device and host memory)
    float* d_embeddings;      // vocab_size x embedding_dim
    float* h_embeddings;      // vocab_size x embedding_dim
    
    // Gradients (device memory)
    float* d_embedding_grads; // vocab_size x embedding_dim
    
    // Adam optimizer first (m) and second (v) moment estimates (device pointers)
    float* d_embedding_m;     // First moment
    float* d_embedding_v;     // Second moment
    
    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
} Embeddings;

// ---------------------------------------------------------------------
// CUDA kernel: Embed input bytes (forward pass)
// ---------------------------------------------------------------------
__global__ void embed_bytes_kernel(float* output, 
                                  const unsigned char* bytes, 
                                  const float* embeddings, 
                                  int batch_size, 
                                  int embedding_dim) {
    int batch_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (batch_idx < batch_size && emb_idx < embedding_dim) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding table
        int embedding_offset = byte_val * embedding_dim;
        
        // Copy the embedding vector to output
        output[batch_idx * embedding_dim + emb_idx] = embeddings[embedding_offset + emb_idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Compute embedding gradients (backward pass)
// ---------------------------------------------------------------------
__global__ void embedding_gradient_kernel(float* embedding_grads,
                                         const float* input_grads,
                                         const unsigned char* bytes,
                                         int batch_size,
                                         int embedding_dim) {
    int batch_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (batch_idx < batch_size && emb_idx < embedding_dim) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding gradient table
        int grad_offset = byte_val * embedding_dim + emb_idx;
        
        // Add gradient to embedding gradient table
        atomicAdd(&embedding_grads[grad_offset], 
                 input_grads[batch_idx * embedding_dim + emb_idx]);
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: AdamW update for embeddings
// ---------------------------------------------------------------------
__global__ void adamw_embeddings_kernel(float* W, const float* grad, float* m, float* v, 
                                       int size, float beta1, float beta2, float epsilon, 
                                       float weight_decay, float learning_rate, int batch_size, 
                                       float bias_correction1, float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Softmax for probabilities output
// ---------------------------------------------------------------------
__global__ void softmax_kernel(float* logits, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
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
// CUDA kernel: Cross-entropy loss computation
// ---------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(float* loss, 
                                         float* d_error, 
                                         const float* probs, 
                                         const float* targets, 
                                         int batch_size, 
                                         int vocab_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* batch_probs = probs + batch_idx * vocab_size;
        const float* batch_target = targets + batch_idx * vocab_size;
        float* batch_error = d_error + batch_idx * vocab_size;
        float batch_loss = 0.0f;
        
        for (int i = 0; i < vocab_size; i++) {
            // Cross-entropy loss contribution: -target * log(prob)
            float prob = fmaxf(batch_probs[i], 1e-15f); // Prevent log(0)
            if (batch_target[i] > 0.0f) {
                batch_loss -= batch_target[i] * logf(prob);
            }
            
            // Gradient for softmax with cross-entropy: prob - target
            batch_error[i] = batch_probs[i] - batch_target[i];
        }
        
        atomicAdd(loss, batch_loss);
    }
}

// ---------------------------------------------------------------------
// Function: Initialize embeddings
// Initializes the embeddings structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
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
    
    // Allocate host memory for embeddings
    emb->h_embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    
    // Initialize matrices with scaled random values
    float scale = 1.0f / sqrtf(embedding_dim);
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        emb->h_embeddings[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
    
    // Allocate device memory for embeddings
    CHECK_CUDA(cudaMalloc(&emb->d_embeddings, vocab_size * embedding_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_grads, vocab_size * embedding_dim * sizeof(float)));
    
    // Allocate device memory for Adam first and second moment estimates and initialize to zero
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_m, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&emb->d_embedding_v, vocab_size * embedding_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMemset(emb->d_embedding_grads, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(emb->d_embedding_m, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(emb->d_embedding_v, 0, vocab_size * embedding_dim * sizeof(float)));
    
    // Copy embeddings from host to device
    CHECK_CUDA(cudaMemcpy(emb->d_embeddings, emb->h_embeddings, 
                         vocab_size * embedding_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    return emb;
}

// ---------------------------------------------------------------------
// Function: Forward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_forward(Embeddings* emb, unsigned char* d_bytes, float* d_output, int batch_size) {
    embed_bytes_kernel<<<batch_size, emb->embedding_dim>>>(
        d_output, d_bytes, emb->d_embeddings, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Backward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_backward(Embeddings* emb, float* d_input_grads, unsigned char* d_bytes, int batch_size) {
    embedding_gradient_kernel<<<batch_size, emb->embedding_dim>>>(
        emb->d_embedding_grads, d_input_grads, d_bytes, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Zero gradients for embeddings
// ---------------------------------------------------------------------
void zero_embedding_gradients(Embeddings* emb) {
    size_t emb_size = emb->vocab_size * emb->embedding_dim * sizeof(float);
    CHECK_CUDA(cudaMemset(emb->d_embedding_grads, 0, emb_size));
}

// ---------------------------------------------------------------------
// Function: Update embeddings with AdamW optimizer
// ---------------------------------------------------------------------
void update_embeddings(Embeddings* emb, float learning_rate, int batch_size) {
    emb->adam_t++; // Increment time step
    
    float bias_correction1 = 1.0f - powf(emb->beta1, (float)emb->adam_t);
    float bias_correction2 = 1.0f - powf(emb->beta2, (float)emb->adam_t);
    
    int block_size = 256;
    int size = emb->vocab_size * emb->embedding_dim;
    int num_blocks = (size + block_size - 1) / block_size;
    
    adamw_embeddings_kernel<<<num_blocks, block_size>>>(
        emb->d_embeddings, emb->d_embedding_grads, emb->d_embedding_m, emb->d_embedding_v,
        size, emb->beta1, emb->beta2, emb->epsilon, emb->weight_decay, 
        learning_rate, batch_size, bias_correction1, bias_correction2);
}

// ---------------------------------------------------------------------
// Function: Free embeddings
// Frees all allocated memory (both device and host).
// ---------------------------------------------------------------------
void free_embeddings(Embeddings* emb) {
    if (!emb) return;
    
    // Free device memory
    cudaFree(emb->d_embeddings);
    cudaFree(emb->d_embedding_grads);
    cudaFree(emb->d_embedding_m);
    cudaFree(emb->d_embedding_v);
    
    // Free host memory
    free(emb->h_embeddings);
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
    
    // Copy embeddings from device to host
    CHECK_CUDA(cudaMemcpy(emb->h_embeddings, emb->d_embeddings, 
                         emb->vocab_size * emb->embedding_dim * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Write embeddings to file
    fwrite(emb->h_embeddings, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    
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
    
    // Read embeddings from file to host memory
    fread(emb->h_embeddings, sizeof(float), vocab_size * embedding_dim, file);
    
    // Copy embeddings to device
    CHECK_CUDA(cudaMemcpy(emb->d_embeddings, emb->h_embeddings, 
                         vocab_size * embedding_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Embeddings loaded from %s\n", filename);
    return emb;
}

// ---------------------------------------------------------------------
// Function: Calculate cross-entropy loss
// ---------------------------------------------------------------------
float calculate_cross_entropy_loss(SSM* ssm, float* d_y) {
    // Apply softmax to get probabilities
    softmax_kernel<<<ssm->batch_size, 1>>>(ssm->d_predictions, 
                                          ssm->batch_size, 
                                          ssm->output_dim);
    
    // Initialize loss to zero
    float h_loss = 0.0f;
    float* d_loss;
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    // Compute cross-entropy loss and gradients
    cross_entropy_loss_kernel<<<ssm->batch_size, 1>>>(d_loss, 
                                                     ssm->d_error, 
                                                     ssm->d_predictions, 
                                                     d_y, 
                                                     ssm->batch_size, 
                                                     ssm->output_dim);
    
    // Copy loss back to host
    CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_loss));
    
    // Return average loss per batch item
    return h_loss / ssm->batch_size;
}

// ---------------------------------------------------------------------
// Function: Reorganize data for batch processing
// (Converts from [example][time] to [time][example] layout)
// ---------------------------------------------------------------------
void reorganize_data(const unsigned char* input, unsigned char* output, 
                    int num_examples, int seq_length) {
    for (int example = 0; example < num_examples; example++) {
        for (int step = 0; step < seq_length; step++) {
            int src_idx = example * seq_length + step;
            int dst_idx = step * num_examples + example;
            
            // Check bounds
            if (src_idx < num_examples * seq_length && dst_idx < num_examples * seq_length) {
                output[dst_idx] = input[src_idx];
            }
        }
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 512;     // Embedding dimension
    int state_dim = 1024;        // State dimension (reduced from 2048)
    int vocab_size = 256;        // One per possible byte value
    float learning_rate = 0.0001; // Learning rate
    int num_epochs = 1000;        // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("Vocabulary size: %d (byte values)\n", vocab_size);
    printf("Embedding dimension: %d\n", embedding_dim);
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("Training epochs: %d\n\n", num_epochs);
    
    // Read training data
    FILE* file = fopen("data.txt", "rb");
    if (!file) {
        fprintf(stderr, "Error opening data.txt\n");
        return EXIT_FAILURE;
    }
    
    // Count number of lines (conversations)
    int num_conversations = 0;
    char buffer[4096];
    
    rewind(file);
    while (fgets(buffer, sizeof(buffer), file)) {
        num_conversations++;
    }
    
    printf("Found %d conversations in data.txt\n", num_conversations);
    
    // Use all conversations for batch processing
    int batch_size = num_conversations;
    printf("Using batch size: %d (processing all conversations in parallel)\n", batch_size);
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read the whole file
    char* text_data = (char*)malloc(file_size + 1);
    if (!text_data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return EXIT_FAILURE;
    }
    
    size_t bytes_read = fread(text_data, 1, file_size, file);
    text_data[bytes_read] = '\0';
    fclose(file);
    
    printf("Loaded %zu bytes of text data\n", bytes_read);
    
    // Split data into conversations (each line is an example)
    char** conversations = (char**)malloc(batch_size * sizeof(char*));
    int conversation_count = 0;
    
    // Load all available conversations
    char* line = strtok(text_data, "\n");
    while (line && conversation_count < batch_size) {
        size_t len = strlen(line);
        if (len > 0) {  // Skip empty lines
            conversations[conversation_count] = (char*)malloc(len + 1);
            strcpy(conversations[conversation_count], line);
            conversation_count++;
        }
        line = strtok(NULL, "\n");
    }
    
    printf("Loaded %d conversations for training\n", conversation_count);
    
    // Find the minimum length among conversations
    int min_length = strlen(conversations[0]);
    for (int i = 1; i < conversation_count; i++) {
        int len = strlen(conversations[i]);
        if (len < min_length) min_length = len;
    }
    
    // Use sequence length of minimum conversation length
    int seq_length = min_length;
    printf("Using sequence length: %d (from shortest conversation)\n", seq_length);
    
    // Prepare input and target data
    unsigned char* h_X_data = (unsigned char*)malloc(conversation_count * seq_length * sizeof(unsigned char));
    unsigned char* h_y_data = (unsigned char*)malloc(conversation_count * seq_length * sizeof(unsigned char));
    
    // Copy data and set up X (current char) and y (next char)
    for (int ex = 0; ex < conversation_count; ex++) {
        for (int pos = 0; pos < seq_length - 1; pos++) {
            h_X_data[ex * seq_length + pos] = conversations[ex][pos];
            h_y_data[ex * seq_length + pos] = conversations[ex][pos + 1];
        }
        // For the last position, use a wrap-around (last char predicts first char)
        h_X_data[ex * seq_length + (seq_length - 1)] = conversations[ex][seq_length - 1];
        h_y_data[ex * seq_length + (seq_length - 1)] = conversations[ex][0];
    }
    
    // Reorganize data to [time][example] layout for efficient processing
    unsigned char* h_X_time_major = (unsigned char*)malloc(conversation_count * seq_length * sizeof(unsigned char));
    unsigned char* h_y_time_major = (unsigned char*)malloc(conversation_count * seq_length * sizeof(unsigned char));
    
    reorganize_data(h_X_data, h_X_time_major, conversation_count, seq_length);
    reorganize_data(h_y_data, h_y_time_major, conversation_count, seq_length);
    
    // Free original arrays
    free(h_X_data);
    free(h_y_data);
    
    // Create one-hot encoded targets
    float* h_y_onehot = (float*)calloc(conversation_count * seq_length * vocab_size, sizeof(float));
    for (int t = 0; t < seq_length; t++) {
        for (int ex = 0; ex < conversation_count; ex++) {
            int idx = t * conversation_count + ex;
            unsigned char target = h_y_time_major[idx];
            h_y_onehot[idx * vocab_size + target] = 1.0f;
        }
    }
    
    // Transfer data to GPU
    unsigned char* d_X_bytes;
    float* d_y_onehot;
    
    CHECK_CUDA(cudaMalloc(&d_X_bytes, conversation_count * seq_length * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_y_onehot, conversation_count * seq_length * vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_X_bytes, h_X_time_major, 
                         conversation_count * seq_length * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_onehot, h_y_onehot, 
                         conversation_count * seq_length * vocab_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize the embedding layer
    Embeddings* embeddings = init_embeddings(vocab_size, embedding_dim);
    
    // Allocate memory for embedded inputs
    float* d_X_embedded;
    CHECK_CUDA(cudaMalloc(&d_X_embedded, conversation_count * embedding_dim * sizeof(float)));
    
    // Initialize SSM model with batch_size = number of conversations
    SSM* ssm = init_ssm(embedding_dim, state_dim, vocab_size, conversation_count);
    
    printf("\nStarting training for %d epochs with batch size of %d...\n", num_epochs, conversation_count);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, conversation_count * state_dim * sizeof(float)));
        
        float epoch_loss = 0.0f;
        
        // Process each timestep
        for (int t = 0; t < seq_length; t++) {
            // Get current timestep data
            unsigned char* d_X_t = d_X_bytes + t * conversation_count;
            float* d_y_t = d_y_onehot + t * conversation_count * vocab_size;
            
            // Forward pass: embedding layer
            embeddings_forward(embeddings, d_X_t, d_X_embedded, conversation_count);
            
            // Forward pass: SSM layer
            forward_pass(ssm, d_X_embedded);
            
            // Calculate loss
            float loss = calculate_cross_entropy_loss(ssm, d_y_t);
            epoch_loss += loss;
            
            // Backward pass: SSM
            zero_gradients(ssm);
            backward_pass(ssm, d_X_embedded);
            
            // Backward pass: embeddings
            zero_embedding_gradients(embeddings);
            embeddings_backward(embeddings, ssm->d_B_grad, d_X_t, conversation_count);
            
            // Update weights
            update_weights(ssm, learning_rate);
            update_embeddings(embeddings, learning_rate, conversation_count);
        }
        
        // Calculate average loss
        float avg_loss = epoch_loss / seq_length;
        
        // Print progress every 10 epochs, but always print first and last
        if (epoch == 0 || epoch == num_epochs - 1 || (epoch + 1) % 10 == 0) {
            printf("Epoch %d/%d, Average Loss: %.6f\n", 
                   epoch + 1, num_epochs, avg_loss);
        }
    }
    
    // Save the final model
    char model_fname[64];
    char embedding_fname[64];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    sprintf(model_fname, "%04d%02d%02d_%02d%02d%02d_slm.bin", 
           timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
           timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
    sprintf(embedding_fname, "%04d%02d%02d_%02d%02d%02d_embeddings.bin", 
           timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
           timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
    
    // For inference, we need batch_size=1
    ssm->batch_size = 1;
    save_ssm(ssm, model_fname);
    save_embeddings(embeddings, embedding_fname);

    // Clean up
    free_ssm(ssm);
    free_embeddings(embeddings);
    
    for (int i = 0; i < conversation_count; i++) {
        free(conversations[i]);
    }
    free(conversations);
    free(text_data);
    free(h_X_time_major);
    free(h_y_time_major);
    free(h_y_onehot);
    
    cudaFree(d_X_bytes);
    cudaFree(d_X_embedded);
    cudaFree(d_y_onehot);
    
    printf("\nTraining completed!\n");
    return 0;
}