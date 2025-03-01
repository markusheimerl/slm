#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"

// Read text file into memory
char* read_text_file(const char* filename, size_t* text_length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);
    
    // Allocate memory for text (add 1 for null terminator)
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Read file into buffer
    *text_length = fread(buffer, 1, file_size, file);
    buffer[*text_length] = '\0'; // Null-terminate
    
    fclose(file);
    return buffer;
}

// Prepare character-level prediction data directly in GPU-friendly layout
void prepare_training_data(const char* text, size_t text_length, 
                          int seq_length, int* num_sequences,
                          float** h_X, float** h_y) {
    // Calculate number of complete sequences we can extract
    *num_sequences = (int)((text_length - 1) / seq_length);
    printf("Creating %d sequences of length %d\n", *num_sequences, seq_length);
    
    // Allocate memory for input and target sequences in GPU-friendly batch format
    // Each timestep contains all sequences in parallel
    size_t data_size = (*num_sequences) * seq_length;
    *h_X = (float*)malloc(data_size * sizeof(float));
    *h_y = (float*)malloc(data_size * sizeof(float));
    
    if (!*h_X || !*h_y) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(EXIT_FAILURE);
    }
    
    // Fill data arrays in batch-oriented layout:
    // [seq0_t0, seq1_t0, ..., seqN_t0, seq0_t1, seq1_t1, ...]
    for (int step = 0; step < seq_length; step++) {
        for (int seq = 0; seq < *num_sequences; seq++) {
            size_t text_idx = seq * seq_length + step;
            int data_idx = step * (*num_sequences) + seq;
            
            // Ensure we don't exceed text bounds
            if (text_idx < text_length && text_idx + 1 < text_length) {
                // Input: current character (normalized)
                (*h_X)[data_idx] = (unsigned char)text[text_idx] / 255.0f;
                
                // Target: next character (normalized)
                (*h_y)[data_idx] = (unsigned char)text[text_idx + 1] / 255.0f;
            }
        }
    }
}

// Verify prepared data for debugging
void verify_data(const char* text, int seq_length, float* X, float* y, int num_sequences) {
    printf("\n=== DATA VERIFICATION ===\n");
    
    // Show original text snippet
    printf("Original text first 30 chars: \"");
    for (int i = 0; i < 30 && text[i] != '\0'; i++) {
        putchar(text[i]);
    }
    printf("...\"\n\n");
    
    // Check sequence 0 preparation
    printf("Sequence 0 (first 10 steps):\n");
    printf("Step\tOrig Char\tX (Input)\tY (Target)\n");
    printf("----------------------------------------\n");
    for (int step = 0; step < 10 && step < seq_length; step++) {
        int data_idx = step * num_sequences + 0;  // First sequence at each step
        size_t text_idx = step;
        char orig_char = text[text_idx];
        char next_char = text[text_idx + 1];
        
        printf("%d\t'%c' (%d)\t%.3f\t\t%.3f ('%c')\n", 
               step, orig_char, (unsigned char)orig_char,
               X[data_idx], y[data_idx], next_char);
    }
    
    // Check sequence 1 preparation if available
    if (num_sequences > 1) {
        printf("\nSequence 1 (first 10 steps):\n");
        printf("Step\tOrig Char\tX (Input)\tY (Target)\n");
        printf("----------------------------------------\n");
        for (int step = 0; step < 10 && step < seq_length; step++) {
            int data_idx = step * num_sequences + 1;  // Second sequence at each step
            size_t text_idx = seq_length + step;  // Start of second sequence
            char orig_char = text[text_idx];
            char next_char = text[text_idx + 1];
            
            printf("%d\t'%c' (%d)\t%.3f\t\t%.3f ('%c')\n", 
                   step, orig_char, (unsigned char)orig_char,
                   X[data_idx], y[data_idx], next_char);
        }
    }
    
    printf("\n=== END DATA VERIFICATION ===\n\n");
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int input_dim = 1;           // Single byte input
    int output_dim = 1;          // Single byte output
    int state_dim = 2048;        // Increased from 512 to 2048
    int seq_length = 128;        // Sequence length for training
    float learning_rate = 0.0002; // Initial learning rate
    float lr_decay = 0.99;       // Learning rate decay per epoch
    int num_epochs = 300;        // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("State dimension: %d\n", state_dim);
    printf("Sequence length: %d\n", seq_length);
    printf("Learning rate: %.6f (decay: %.3f per epoch)\n", learning_rate, lr_decay);
    printf("Training epochs: %d\n\n", num_epochs);
    
    // Read training data
    size_t text_length;
    char* text = read_text_file("data.txt", &text_length);
    printf("Loaded %zu characters of text data\n", text_length);
    
    // Print a small sample of the text
    printf("\nText sample:\n------------------\n");
    size_t sample_size = text_length < 100 ? text_length : 100;
    for (size_t i = 0; i < sample_size; i++) {
        putchar(text[i]);
    }
    printf("\n------------------\n\n");
    
    // Prepare training data in batch-friendly format
    int num_sequences;
    float *h_X, *h_y;
    prepare_training_data(text, text_length, seq_length, &num_sequences, &h_X, &h_y);
    
    // Verify prepared data
    verify_data(text, seq_length, h_X, h_y, num_sequences);
    
    // Transfer data to GPU directly - already in the right format
    float *d_X, *d_y;
    size_t data_size = num_sequences * seq_length * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_X, data_size));
    CHECK_CUDA(cudaMalloc(&d_y, data_size));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, data_size, cudaMemcpyHostToDevice));
    
    // Initialize model (each sequence is processed in parallel)
    int batch_size = num_sequences;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    printf("Starting training for %d epochs (batch size: %d sequences)\n", 
           num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        float current_lr = learning_rate * powf(lr_decay, epoch);
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        // Process each timestep across all sequences
        for (int step = 0; step < seq_length; step++) {
            // Get step data (one character from each sequence at this timestep)
            // Data is already organized in batch format
            float* step_X_ptr = &d_X[step * batch_size];
            float* step_y_ptr = &d_y[step * batch_size];
            
            // Forward pass
            forward_pass(ssm, step_X_ptr);
            
            // Calculate loss
            float step_loss = calculate_loss(ssm, step_y_ptr);
            epoch_loss += step_loss;
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, step_X_ptr);
            update_weights(ssm, current_lr);
            
            // Print progress periodically
            if (step % 20 == 0) {
                printf("\rEpoch %d/%d, Step %d/%d, Loss: %.6f, LR: %.6f", 
                       epoch + 1, num_epochs, step + 1, seq_length, step_loss, current_lr);
                fflush(stdout);
            }
        }
        
        // Print epoch statistics
        epoch_loss /= seq_length;
        printf("\nEpoch %d/%d completed, Average Loss: %.6f\n", 
               epoch + 1, num_epochs, epoch_loss);
        
        // Save model checkpoint at the end
        if (epoch == num_epochs - 1) {
            // Save final model
            char model_fname[64];
            time_t now = time(NULL);
            struct tm *timeinfo = localtime(&now);
            sprintf(model_fname, "%04d%02d%02d_%02d%02d%02d_slm_final.bin", 
                   timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
                   timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
            
            // Create inference model with batch_size=1 for saving
            SSM* inference_model = init_ssm(input_dim, state_dim, output_dim, 1);
            CHECK_CUDA(cudaMemcpy(inference_model->d_A, ssm->d_A, 
                                 state_dim * state_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_model->d_B, ssm->d_B, 
                                 state_dim * input_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_model->d_C, ssm->d_C, 
                                 output_dim * state_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_model->d_D, ssm->d_D, 
                                 output_dim * input_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            save_ssm(inference_model, model_fname);
            printf("Model saved to %s\n", model_fname);
            
            free_ssm(inference_model);
        }
    }
    
    printf("\nTraining completed!\n");
    
    // Clean up
    free(text);
    free(h_X);
    free(h_y);
    cudaFree(d_X);
    cudaFree(d_y);
    free_ssm(ssm);
    
    return 0;
}