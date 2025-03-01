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

// Split text into separate training examples (conversations)
char** split_into_examples(const char* text, int* num_examples) {
    // Count the number of examples (separated by single newlines)
    int count = 0;
    const char* pos = text;
    while ((pos = strstr(pos, "\n")) != NULL) {
        count++;
        pos++;
    }
    
    // Allocate array for example pointers
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

// Prepare character-level prediction data directly in GPU-friendly layout
void prepare_training_data(char** examples, int num_examples, int seq_length,
                           float** h_X, float** h_y) {
    printf("Preparing training data: %d examples, sequence length: %d\n", 
           num_examples, seq_length);
    
    // Allocate memory for input and target sequences in GPU-friendly batch format
    // Each timestep contains all examples in parallel
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

// Detailed verification of the prepared data
void verify_data(char** examples, int seq_length, float* X, float* y, int num_examples) {
    printf("\n=== DETAILED DATA VERIFICATION ===\n");
    
    // Print basic info
    printf("Number of examples: %d\n", num_examples);
    printf("Sequence length: %d\n", seq_length);
    printf("Total data points: %d\n\n", num_examples * seq_length);
    
    // Show first few characters of each example
    printf("Sample of training examples:\n");
    printf("----------------------------\n");
    for (int ex = 0; ex < num_examples && ex < 5; ex++) {
        int preview_len = 50;
        char preview[51];  // +1 for null terminator
        
        strncpy(preview, examples[ex], 50);
        preview[50] = '\0';
        
        printf("Example %2d: \"%s%s\"\n", 
               ex, preview, strlen(examples[ex]) > 50 ? "..." : "");
    }
    printf("----------------------------\n\n");
    
    // Visualize the batching structure
    printf("Batching structure verification:\n");
    printf("-------------------------------\n");
    printf("Each timestep processes one character from each example in parallel.\n");
    printf("Data layout: [ex0_t0, ex1_t0, ..., exN_t0, ex0_t1, ex1_t1, ...]\n\n");
    
    // Show detailed timestep data for verification
    int rows_to_show = 5;  // Show first 5 examples
    int cols_to_show = 10; // Show first 10 timesteps
    
    printf("Input Data (X) - First %d timesteps x %d examples:\n", cols_to_show, rows_to_show);
    printf("Step\t");
    for (int ex = 0; ex < rows_to_show && ex < num_examples; ex++) {
        printf("Ex%d\t\t", ex);
    }
    printf("\n");
    
    for (int step = 0; step < cols_to_show && step < seq_length; step++) {
        printf("%d\t", step);
        for (int ex = 0; ex < rows_to_show && ex < num_examples; ex++) {
            int data_idx = step * num_examples + ex;
            char c = (int)(X[data_idx] * 255.0f);
            printf("'%c'(%.3f)\t", c, X[data_idx]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Show target data
    printf("Target Data (y) - First %d timesteps x %d examples:\n", cols_to_show, rows_to_show);
    printf("Step\t");
    for (int ex = 0; ex < rows_to_show && ex < num_examples; ex++) {
        printf("Ex%d\t\t", ex);
    }
    printf("\n");
    
    for (int step = 0; step < cols_to_show && step < seq_length; step++) {
        printf("%d\t", step);
        for (int ex = 0; ex < rows_to_show && ex < num_examples; ex++) {
            int data_idx = step * num_examples + ex;
            char c = (int)(y[data_idx] * 255.0f);
            printf("'%c'(%.3f)\t", c, y[data_idx]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Verify a specific sequence
    if (num_examples >= 2) {
        int ex_to_verify = 1;  // Verify the second example
        printf("Sequence verification for Example %d:\n", ex_to_verify);
        printf("Original text: \"%.*s...\"\n", 30, examples[ex_to_verify]);
        printf("Step\tX (Input)\t\tY (Target)\t\tOriginal\tExpected Next\n");
        printf("---------------------------------------------------------------------------\n");
        
        for (int step = 0; step < 20 && step < seq_length - 1; step++) {
            int data_idx = step * num_examples + ex_to_verify;
            char orig_char = examples[ex_to_verify][step];
            char next_char = examples[ex_to_verify][step + 1];
            
            float x_val = X[data_idx];
            float y_val = y[data_idx];
            
            char x_char = (char)(x_val * 255.0f + 0.5f);
            char y_char = (char)(y_val * 255.0f + 0.5f);
            
            printf("%d\t'%c'(%.3f)\t\t'%c'(%.3f)\t\t'%c'(%d)\t\t'%c'(%d)\n", 
                   step, x_char, x_val, y_char, y_val, 
                   orig_char, (unsigned char)orig_char, 
                   next_char, (unsigned char)next_char);
        }
    }
    
    // Verify batch consistency - check if characters line up with original text
    printf("\nBatch consistency verification:\n");
    printf("Checking if batch data matches original text at different positions...\n");
    
    int positions_to_check[] = {0, 5, 10, 20, 50, 100};
    int num_positions = sizeof(positions_to_check) / sizeof(positions_to_check[0]);
    
    for (int i = 0; i < num_positions; i++) {
        int pos = positions_to_check[i];
        if (pos >= seq_length) continue;
        
        printf("Position %d across examples:\n", pos);
        for (int ex = 0; ex < 5 && ex < num_examples; ex++) {
            int data_idx = pos * num_examples + ex;
            char original = examples[ex][pos];
            float x_val = X[data_idx];
            char x_char = (char)(x_val * 255.0f + 0.5f);
            
            if (original == x_char) {
                printf("Example %d: Original '%c' matches data '%c' ✓\n", 
                       ex, original, x_char);
            } else {
                printf("Example %d: Original '%c' (%d) DOES NOT MATCH data '%c' (%d) ✗\n", 
                       ex, original, (unsigned char)original, x_char, (unsigned char)x_char);
            }
        }
        printf("\n");
    }
    
    printf("=== END DATA VERIFICATION ===\n\n");
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int input_dim = 1;            // Single byte input
    int output_dim = 1;           // Single byte output
    int state_dim = 2048;         // State dimension
    float learning_rate = 0.0002; // Fixed learning rate without decay
    int num_epochs = 300;         // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.6f (fixed, no decay)\n", learning_rate);
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
    
    // Split text into separate training examples
    int num_examples;
    char** examples = split_into_examples(text, &num_examples);
    printf("Split data into %d separate examples\n", num_examples);
    
    // Find the minimum length for sequence processing
    int min_length = find_min_length(examples, num_examples);
    int seq_length = min_length > 512 ? 512 : min_length; // Cap at 512 chars if needed
    printf("Minimum example length: %d, Using sequence length: %d\n", min_length, seq_length);
    
    // Prepare training data in batch-friendly format
    float *h_X, *h_y;
    prepare_training_data(examples, num_examples, seq_length, &h_X, &h_y);
    
    // Verify prepared data with detailed output
    verify_data(examples, seq_length, h_X, h_y, num_examples);
    
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
    
    printf("Starting training for %d epochs (batch size: %d examples)\n", 
           num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
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
            epoch_loss += step_loss;
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, step_X_ptr);
            update_weights(ssm, learning_rate);
            
            // Print progress periodically 
            if (step % 20 == 0 || step == seq_length - 1) {
                printf("\rEpoch %d/%d, Step %d/%d, Loss: %.6f", 
                       epoch + 1, num_epochs, step + 1, seq_length, step_loss);
                fflush(stdout);
            }
        }
        
        // Print epoch statistics
        epoch_loss /= seq_length;
        printf("\nEpoch %d/%d completed, Average Loss: %.6f\n", 
               epoch + 1, num_epochs, epoch_loss);
        
        // Save model checkpoint periodically or at the end
        if ((epoch + 1) % 50 == 0 || epoch == num_epochs - 1) {
            char model_fname[64];
            time_t now = time(NULL);
            struct tm *timeinfo = localtime(&now);
            sprintf(model_fname, "%04d%02d%02d_%02d%02d%02d_slm_epoch%d.bin", 
                   timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
                   timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, 
                   epoch + 1);
            
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