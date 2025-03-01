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

// Prepare character-level prediction data in GPU-friendly layout
void prepare_training_data(char** examples, int num_examples, int seq_length,
                           float** h_X, float** h_y) {
    size_t data_size = num_examples * seq_length;
    *h_X = (float*)malloc(data_size * sizeof(float));
    *h_y = (float*)malloc(data_size * sizeof(float));
    
    if (!*h_X || !*h_y) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(EXIT_FAILURE);
    }
    
    // Fill data arrays in batch-oriented layout:
    // [ex0_t0, ex1_t0, ..., exN_t0, ex0_t1, ex1_t1, ...]
    for (int step = 0; step < seq_length - 1; step++) {
        for (int ex = 0; ex < num_examples; ex++) {
            int data_idx = step * num_examples + ex;
            
            // Input: current character (normalized)
            (*h_X)[data_idx] = (unsigned char)examples[ex][step] / 255.0f;
            
            // Target: next character (normalized)
            (*h_y)[data_idx] = (unsigned char)examples[ex][step + 1] / 255.0f;
        }
    }
    
    // Handle the last character of each sequence
    int last_step = seq_length - 1;
    for (int ex = 0; ex < num_examples; ex++) {
        int data_idx = last_step * num_examples + ex;
        
        // Input: last character
        (*h_X)[data_idx] = (unsigned char)examples[ex][last_step] / 255.0f;
        
        // Target: we'll use a zero value for simplicity (end of sequence marker)
        (*h_y)[data_idx] = 0.0f;
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int input_dim = 1;            // Single byte input
    int output_dim = 1;           // Single byte output
    int state_dim = 2048;         // State dimension
    float learning_rate = 0.0002; // Fixed learning rate
    int num_epochs = 300;         // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.6f\n", learning_rate);
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
    
    // Prepare training data in batch-friendly format
    float *h_X, *h_y;
    prepare_training_data(examples, num_examples, seq_length, &h_X, &h_y);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    size_t data_size = num_examples * seq_length * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_X, data_size));
    CHECK_CUDA(cudaMalloc(&d_y, data_size));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, data_size, cudaMemcpyHostToDevice));
    
    // Initialize model (each example is processed in parallel)
    int batch_size = num_examples;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    printf("Starting training for %d epochs (batch size: %d examples)\n\n", 
           num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        // Process each timestep across all examples
        for (int step = 0; step < seq_length; step++) {
            // Get step data (one character from each example at this timestep)
            float* step_X_ptr = &d_X[step * batch_size];
            float* step_y_ptr = &d_y[step * batch_size];
            
            // Forward pass
            forward_pass(ssm, step_X_ptr);
            
            // Calculate loss
            float step_loss = calculate_loss(ssm, step_y_ptr);
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, step_X_ptr);
            update_weights(ssm, learning_rate);
            
            if (step == seq_length - 1) {
                printf("Epoch %d/%d, Loss: %.6f\n", 
                       epoch + 1, num_epochs, step_loss);
            }
        }
    }
    
    // Save the final model
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
    printf("\nModel saved to %s\n", model_fname);
    
    free_ssm(inference_model);
    printf("Training completed!\n");
    
    // Clean up
    for (int i = 0; i < num_examples; i++) {
        free(examples[i]);
    }
    free(examples);
    free(text);
    free(h_X);
    free(h_y);
    cudaFree(d_X);
    cudaFree(d_y);
    free_ssm(ssm);
    
    return 0;
}