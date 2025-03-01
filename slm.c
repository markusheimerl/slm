#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <ctype.h>
#include "ssm/gpu/ssm.h"

#define MAX_TEXT_LENGTH 10000000 // 10MB buffer for text
#define VOCAB_SIZE 256 // ASCII characters
#define SEQUENCE_LENGTH 64 // Characters per training sequence
#define BATCH_SIZE 32 // Batch size for training
#define STATE_DIM 512 // Hidden state dimension
#define EPOCHS 20 // Number of training epochs
#define LEARNING_RATE 0.0001f // Learning rate for Adam optimizer

// Function to read text file into memory
char* read_text_file(const char* filename, size_t* text_length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    char* buffer = (char*)malloc(MAX_TEXT_LENGTH);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    *text_length = fread(buffer, 1, MAX_TEXT_LENGTH - 1, file);
    buffer[*text_length] = '\0'; // Null-terminate the string
    
    fclose(file);
    return buffer;
}

// Convert text to one-hot encoding
void text_to_one_hot(const char* text, size_t text_length, float* one_hot, int sequence_length) {
    for (size_t i = 0; i < text_length - 1 && i < sequence_length; i++) {
        unsigned char c = (unsigned char)text[i];
        unsigned char next_c = (unsigned char)text[i + 1];
        
        // Set input (current character one-hot vector)
        memset(&one_hot[i * VOCAB_SIZE], 0, VOCAB_SIZE * sizeof(float));
        one_hot[i * VOCAB_SIZE + c] = 1.0f;
        
        // Set target (next character one-hot vector)
        memset(&one_hot[(i + sequence_length) * VOCAB_SIZE], 0, VOCAB_SIZE * sizeof(float));
        one_hot[(i + sequence_length) * VOCAB_SIZE + next_c] = 1.0f;
    }
}

// Create batches from text data
void create_batch(const char* text, size_t text_length, int batch_idx, float* batch_X, float* batch_y) {
    int offset = batch_idx * SEQUENCE_LENGTH * BATCH_SIZE;
    
    // Check if we have enough data left
    if (offset + SEQUENCE_LENGTH * BATCH_SIZE >= text_length) {
        fprintf(stderr, "Not enough data for batch %d\n", batch_idx);
        return;
    }
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        int seq_start = offset + i * SEQUENCE_LENGTH;
        
        // One-hot encode input (X) and target (y)
        for (int j = 0; j < SEQUENCE_LENGTH; j++) {
            // Skip if we're out of bounds
            if (seq_start + j >= text_length - 1) break;
            
            unsigned char c = (unsigned char)text[seq_start + j];
            unsigned char next_c = (unsigned char)text[seq_start + j + 1];
            
            // Input: current character one-hot
            memset(&batch_X[(i * SEQUENCE_LENGTH + j) * VOCAB_SIZE], 0, VOCAB_SIZE * sizeof(float));
            batch_X[(i * SEQUENCE_LENGTH + j) * VOCAB_SIZE + c] = 1.0f;
            
            // Target: next character one-hot
            memset(&batch_y[(i * SEQUENCE_LENGTH + j) * VOCAB_SIZE], 0, VOCAB_SIZE * sizeof(float));
            batch_y[(i * SEQUENCE_LENGTH + j) * VOCAB_SIZE + next_c] = 1.0f;
        }
    }
}

// Generate text from the trained model
void generate_text(SSM* ssm, const char* seed, int length) {
    int seed_length = strlen(seed);
    char* generated = (char*)malloc(seed_length + length + 1);
    if (!generated) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    // Copy seed to the generated text
    strcpy(generated, seed);
    
    // Reset the model state
    CHECK_CUDA(cudaMemset(ssm->d_state, 0, ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    // Allocate device memory for input
    float* h_input = (float*)malloc(VOCAB_SIZE * sizeof(float));
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, VOCAB_SIZE * sizeof(float)));
    
    // Process the seed text to establish state
    for (int i = 0; i < seed_length; i++) {
        // Prepare one-hot input for the current character
        memset(h_input, 0, VOCAB_SIZE * sizeof(float));
        h_input[(unsigned char)seed[i]] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass(ssm, d_input);
    }
    
    // Generate new characters
    for (int i = 0; i < length; i++) {
        // Get predictions from the model
        float* h_predictions = (float*)malloc(VOCAB_SIZE * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_predictions, ssm->d_predictions, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Find the highest probability character
        int max_idx = 0;
        float max_prob = h_predictions[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (h_predictions[j] > max_prob) {
                max_prob = h_predictions[j];
                max_idx = j;
            }
        }
        
        // Add the predicted character to the generated text
        generated[seed_length + i] = (char)max_idx;
        
        // Prepare next input
        memset(h_input, 0, VOCAB_SIZE * sizeof(float));
        h_input[max_idx] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        
        // Forward pass for next prediction
        forward_pass(ssm, d_input);
        
        free(h_predictions);
    }
    
    // Null-terminate the generated string
    generated[seed_length + length] = '\0';
    
    printf("\nGenerated text:\n%s\n", generated);
    
    // Clean up
    free(generated);
    free(h_input);
    cudaFree(d_input);
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Read training data
    size_t text_length;
    char* text = read_text_file("data.txt", &text_length);
    printf("Loaded %zu characters of text data\n", text_length);
    
    // Calculate number of batches
    int num_batches = text_length / (SEQUENCE_LENGTH * BATCH_SIZE);
    printf("Creating %d batches for training\n", num_batches);
    
    // Initialize SSM model
    SSM* ssm = init_ssm(VOCAB_SIZE, STATE_DIM, VOCAB_SIZE, BATCH_SIZE);
    
    // Allocate memory for batches
    float* h_batch_X = (float*)malloc(BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    float* h_batch_y = (float*)malloc(BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    
    // Allocate device memory
    float* d_batch_X, *d_batch_y;
    CHECK_CUDA(cudaMalloc(&d_batch_X, BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_batch_y, BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float)));
    
    // Training loop
    printf("Starting training for %d epochs\n", EPOCHS);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Create batch from text data
            create_batch(text, text_length, batch, h_batch_X, h_batch_y);
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_batch_X, h_batch_X, 
                                 BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float), 
                                 cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_batch_y, h_batch_y, 
                                 BATCH_SIZE * SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float), 
                                 cudaMemcpyHostToDevice));
            
            // Reset state for each sequence
            CHECK_CUDA(cudaMemset(ssm->d_state, 0, ssm->batch_size * ssm->state_dim * sizeof(float)));
            
            float batch_loss = 0.0f;
            
            // Process each step in the sequence
            for (int step = 0; step < SEQUENCE_LENGTH; step++) {
                // Get current step input and target
                float* d_step_X = d_batch_X + step * BATCH_SIZE * VOCAB_SIZE;
                float* d_step_y = d_batch_y + step * BATCH_SIZE * VOCAB_SIZE;
                
                // Forward pass
                forward_pass(ssm, d_step_X);
                
                // Calculate loss
                float step_loss = calculate_loss(ssm, d_step_y);
                batch_loss += step_loss;
                
                // Backward pass
                zero_gradients(ssm);
                backward_pass(ssm, d_step_X);
                
                // Update weights
                update_weights(ssm, LEARNING_RATE);
            }
            
            // Average loss over sequence steps
            batch_loss /= SEQUENCE_LENGTH;
            epoch_loss += batch_loss;
            
            // Print progress
            if (batch % 10 == 0) {
                printf("\rEpoch %d/%d, Batch %d/%d, Loss: %.6f", 
                       epoch + 1, EPOCHS, batch + 1, num_batches, batch_loss);
                fflush(stdout);
            }
        }
        
        // Print epoch statistics
        epoch_loss /= num_batches;
        printf("\nEpoch %d/%d completed, Average Loss: %.6f\n", 
               epoch + 1, EPOCHS, epoch_loss);
        
        // Generate sample text periodically
        if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
            const char* seed = "<USER> ";
            generate_text(ssm, seed, 100);
        }
    }
    
    printf("Training completed!\n");
    
    // Save the trained model
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_ssm(ssm, model_fname);
    printf("Model saved to %s\n", model_fname);
    
    // Clean up
    free(text);
    free(h_batch_X);
    free(h_batch_y);
    cudaFree(d_batch_X);
    cudaFree(d_batch_y);
    free_ssm(ssm);
    
    return 0;
}