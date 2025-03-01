#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"

// ---------------------------------------------------------------------
// CUDA kernel: Softmax activation for cross-entropy loss
// ---------------------------------------------------------------------
__global__ void softmax_kernel(float* predictions, int batch_size, int output_dim) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointer to this batch item's prediction vector
        float* batch_pred = predictions + batch_idx * output_dim;
        
        // Find max value for numerical stability
        float max_val = batch_pred[0];
        for (int i = 1; i < output_dim; i++) {
            max_val = fmaxf(max_val, batch_pred[i]);
        }
        
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < output_dim; i++) {
            batch_pred[i] = expf(batch_pred[i] - max_val);
            sum_exp += batch_pred[i];
        }
        
        // Normalize to get softmax probabilities
        for (int i = 0; i < output_dim; i++) {
            batch_pred[i] /= sum_exp;
        }
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Cross-entropy loss computation
// ---------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(float* loss, float* d_error, 
                                         const float* predictions, 
                                         const float* targets, 
                                         int batch_size, int output_dim) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        const float* batch_pred = predictions + batch_idx * output_dim;
        const float* batch_target = targets + batch_idx * output_dim;
        float* batch_error = d_error + batch_idx * output_dim;
        float batch_loss = 0.0f;
        
        for (int i = 0; i < output_dim; i++) {
            // Cross-entropy loss contribution: -target * log(pred)
            float pred = fmaxf(batch_pred[i], 1e-15f); // Prevent log(0)
            float target = batch_target[i];
            
            if (target > 0.0f) {
                batch_loss -= target * logf(pred);
            }
            
            // Gradient for cross-entropy with softmax is simple: pred - target
            batch_error[i] = batch_pred[i] - target;
        }
        
        atomicAdd(loss, batch_loss);
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Initialize fixed random embeddings
// ---------------------------------------------------------------------
__global__ void initialize_embeddings_kernel(float* embeddings, int num_embeddings, int embedding_dim, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_embeddings * embedding_dim) {
        // Simple random number generator
        unsigned int state = seed + idx;
        state = ((state * 1103515245) + 12345) & 0x7fffffff;
        
        // Scale to [-0.1, 0.1]
        float scale = 0.1f;
        float value = ((float)state / (float)0x7fffffff) * 2.0f - 1.0f;
        embeddings[idx] = value * scale;
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Convert bytes to embeddings in batch
// ---------------------------------------------------------------------
__global__ void bytes_to_embeddings_kernel(float* output, const unsigned char* bytes, 
                                          const float* embeddings, 
                                          int batch_size, int embedding_dim) {
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
// Function: calculate_cross_entropy_loss
// Computes cross-entropy loss between predictions and targets
// ---------------------------------------------------------------------
float calculate_cross_entropy_loss(SSM* ssm, float* d_y) {
    // Apply softmax to predictions
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
    
    // Return average loss per batch
    return h_loss / ssm->batch_size;
}

// Read text file into memory
char* read_text_file(const char* filename, size_t* text_length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);
    
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    *text_length = fread(buffer, 1, file_size, file);
    buffer[*text_length] = '\0';
    
    fclose(file);
    return buffer;
}

// Split text into separate training examples (conversations)
char** split_into_examples(const char* text, int* num_examples) {
    // Count the number of examples (separated by single newlines)
    int count = 0;
    const char* pos = text;
    while ((pos = strstr(pos, "\n")) != NULL) {
        count++;
        pos++;
    }
    
    char** examples = (char**)malloc((count + 1) * sizeof(char*));
    if (!examples) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Split text into examples
    const char* start = text;
    const char* end;
    int idx = 0;
    
    while ((end = strchr(start, '\n')) != NULL) {
        size_t len = end - start;
        examples[idx] = (char*)malloc(len + 1);
        if (!examples[idx]) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        
        strncpy(examples[idx], start, len);
        examples[idx][len] = '\0';
        
        start = end + 1;  // Skip the newline
        idx++;
    }
    
    // Get the last example if it doesn't end with a newline
    if (*start != '\0') {
        size_t len = strlen(start);
        examples[idx] = (char*)malloc(len + 1);
        if (!examples[idx]) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        
        strcpy(examples[idx], start);
        idx++;
    }
    
    *num_examples = idx;
    return examples;
}

// Find the length of the shortest example
int find_min_length(char** examples, int num_examples) {
    if (num_examples == 0) return 0;
    
    int min_len = strlen(examples[0]);
    for (int i = 1; i < num_examples; i++) {
        int len = strlen(examples[i]);
        if (len < min_len) min_len = len;
    }
    
    return min_len;
}

// Prepare byte data in GPU-friendly layout
void prepare_training_data(char** examples, int num_examples, int seq_length,
                          unsigned char** h_X_bytes, float** h_y) {
    // Allocate space for byte input and one-hot output
    *h_X_bytes = (unsigned char*)malloc(num_examples * seq_length * sizeof(unsigned char));
    size_t y_size = num_examples * seq_length * 256;  // One-hot encoded targets
    *h_y = (float*)calloc(y_size, sizeof(float));  // Initialize to all zeros
    
    if (!*h_X_bytes || !*h_y) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(EXIT_FAILURE);
    }
    
    // Fill data arrays
    for (int step = 0; step < seq_length - 1; step++) {
        for (int ex = 0; ex < num_examples; ex++) {
            // Input byte at current position
            unsigned char cur_char = examples[ex][step];
            
            // Calculate position in flattened array
            size_t byte_pos = step * num_examples + ex;
            (*h_X_bytes)[byte_pos] = cur_char;
            
            // Target character (next position)
            unsigned char next_char = examples[ex][step + 1];
            
            // Calculate position in flattened array for target one-hot vector
            size_t y_pos = (byte_pos * 256) + next_char;
            
            // Set the corresponding bit to 1 in the one-hot vector
            (*h_y)[y_pos] = 1.0f;
        }
    }
    
    // Handle the last character of each sequence
    int last_step = seq_length - 1;
    for (int ex = 0; ex < num_examples; ex++) {
        // Input: last character
        unsigned char last_char = examples[ex][last_step];
        size_t byte_pos = last_step * num_examples + ex;
        (*h_X_bytes)[byte_pos] = last_char;
        
        // Target: we'll use a zero vector for simplicity (end of sequence marker)
        // This is already handled by calloc initialization
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 1024;     // Fixed embedding dimension
    int output_dim = 256;         // One-hot encoded output prediction
    int state_dim = 2048;         // State dimension
    float learning_rate = 0.0001; // Fixed learning rate
    int num_epochs = 300;         // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("Embedding dimension: %d (fixed random)\n", embedding_dim);
    printf("Output dimension: %d (one-hot encoded)\n", output_dim);
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.9f\n", learning_rate);
    printf("Training epochs: %d\n\n", num_epochs);
    
    // Read training data
    size_t text_length;
    char* text = read_text_file("data.txt", &text_length);
    printf("Loaded %zu characters of text data\n", text_length);
    
    // Split text into separate training examples
    int num_examples;
    char** examples = split_into_examples(text, &num_examples);
    printf("Split data into %d separate examples\n", num_examples);
    
    // Use the actual length of the shortest example
    int seq_length = find_min_length(examples, num_examples);
    printf("Training with sequence length: %d (from shortest example)\n\n", seq_length);
    
    // Prepare training data
    unsigned char *h_X_bytes;
    float *h_y;
    prepare_training_data(examples, num_examples, seq_length, &h_X_bytes, &h_y);
    
    // Initialize fixed random embeddings
    float *h_embeddings = (float*)malloc(256 * embedding_dim * sizeof(float));
    if (!h_embeddings) {
        fprintf(stderr, "Memory allocation failed for embeddings\n");
        exit(EXIT_FAILURE);
    }
    
    // Generate random embeddings in host memory
    for (int i = 0; i < 256 * embedding_dim; i++) {
        h_embeddings[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;  // Range: [-0.1, 0.1]
    }
    
    // Transfer data to GPU
    unsigned char *d_X_bytes;
    float *d_embeddings, *d_y, *d_X_embedded;
    size_t x_bytes_size = num_examples * seq_length * sizeof(unsigned char);
    size_t embeddings_size = 256 * embedding_dim * sizeof(float);
    size_t y_data_size = num_examples * seq_length * output_dim * sizeof(float);
    size_t x_embedded_size = num_examples * seq_length * embedding_dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_X_bytes, x_bytes_size));
    CHECK_CUDA(cudaMalloc(&d_embeddings, embeddings_size));
    CHECK_CUDA(cudaMalloc(&d_X_embedded, x_embedded_size));
    CHECK_CUDA(cudaMalloc(&d_y, y_data_size));
    
    CHECK_CUDA(cudaMemcpy(d_X_bytes, h_X_bytes, x_bytes_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_embeddings, h_embeddings, embeddings_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, y_data_size, cudaMemcpyHostToDevice));
    
    // Initialize fixed embeddings on GPU with more diversity
    int block_size = 256;
    int num_blocks = (256 * embedding_dim + block_size - 1) / block_size;
    initialize_embeddings_kernel<<<num_blocks, block_size>>>(d_embeddings, 256, embedding_dim, time(NULL));
    
    // Initialize model (each example is processed in parallel)
    int batch_size = num_examples;
    SSM* ssm = init_ssm(embedding_dim, state_dim, output_dim, batch_size);
    
    printf("Starting training for %d epochs (batch size: %d examples)\n\n", 
           num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        float epoch_loss = 0.0f;
        
        // Process each timestep across all examples
        for (int step = 0; step < seq_length; step++) {
            // Get pointers to this step's data across all examples
            unsigned char* step_X_bytes_ptr = &d_X_bytes[step * batch_size];
            float* step_y_ptr = &d_y[step * batch_size * output_dim];
            float* step_X_embedded_ptr = &d_X_embedded[step * batch_size * embedding_dim];
            
            // Convert bytes to embeddings for this batch
            bytes_to_embeddings_kernel<<<batch_size, embedding_dim>>>(
                step_X_embedded_ptr, step_X_bytes_ptr, d_embeddings, batch_size, embedding_dim);
            
            // Forward pass with embedded input
            forward_pass(ssm, step_X_embedded_ptr);
            
            // Calculate cross-entropy loss
            float step_loss = calculate_cross_entropy_loss(ssm, step_y_ptr);
            epoch_loss += step_loss;
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, step_X_embedded_ptr);
            update_weights(ssm, learning_rate);
        }
        
        printf("Epoch %d/%d, Average Loss: %.6f\n", 
               epoch + 1, num_epochs, epoch_loss / seq_length);
    }
    
    // Save the final model
    char model_fname[64];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    sprintf(model_fname, "%04d%02d%02d_%02d%02d%02d_slm_final.bin", 
           timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
           timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
    
    // Create inference model with batch_size=1 for saving
    SSM* inference_model = init_ssm(embedding_dim, state_dim, output_dim, 1);
    CHECK_CUDA(cudaMemcpy(inference_model->d_A, ssm->d_A, 
                         state_dim * state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(inference_model->d_B, ssm->d_B, 
                         state_dim * embedding_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(inference_model->d_C, ssm->d_C, 
                         output_dim * state_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(inference_model->d_D, ssm->d_D, 
                         output_dim * embedding_dim * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // Also save embeddings with the model for inference
    // Note: We need to extend the save_ssm and load_ssm functions to include embeddings
    // For now, we'll just mention this requirement
    printf("\nNote: For full inference functionality, embeddings should also be saved\n");
    
    save_ssm(inference_model, model_fname);
    printf("\nModel saved to %s\n", model_fname);
    
    free_ssm(inference_model);
    printf("Training completed!\n");
    
    // Clean up
    for (int i = 0; i < num_examples; i++) {
        free(examples[i]);
    }
    free(examples);
    free(text);
    free(h_X_bytes);
    free(h_y);
    free(h_embeddings);
    cudaFree(d_X_bytes);
    cudaFree(d_embeddings);
    cudaFree(d_X_embedded);
    cudaFree(d_y);
    free_ssm(ssm);
    
    return 0;
}