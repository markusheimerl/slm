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

// Prepare one-hot encoded character data in GPU-friendly layout
void prepare_training_data(char** examples, int num_examples, int seq_length,
                          float** h_X, float** h_y) {
    // One-hot encoding data size (256 dimensions for each character)
    size_t onehot_dim = 256;
    size_t x_size = num_examples * seq_length * onehot_dim;
    size_t y_size = num_examples * seq_length * onehot_dim;
    
    *h_X = (float*)calloc(x_size, sizeof(float));  // Initialize to all zeros
    *h_y = (float*)calloc(y_size, sizeof(float));  // Initialize to all zeros
    
    if (!*h_X || !*h_y) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(EXIT_FAILURE);
    }
    
    // Fill one-hot encoded data arrays
    for (int step = 0; step < seq_length - 1; step++) {
        for (int ex = 0; ex < num_examples; ex++) {
            // Input character at current position
            unsigned char cur_char = examples[ex][step];
            
            // Calculate position in flattened array for this one-hot vector
            size_t x_pos = (step * num_examples + ex) * onehot_dim + cur_char;
            
            // Set the corresponding bit to 1 in the one-hot vector
            (*h_X)[x_pos] = 1.0f;
            
            // Target character (next position)
            unsigned char next_char = examples[ex][step + 1];
            
            // Calculate position in flattened array for target one-hot vector
            size_t y_pos = (step * num_examples + ex) * onehot_dim + next_char;
            
            // Set the corresponding bit to 1 in the one-hot vector
            (*h_y)[y_pos] = 1.0f;
        }
    }
    
    // Handle the last character of each sequence
    int last_step = seq_length - 1;
    for (int ex = 0; ex < num_examples; ex++) {
        // Input: last character
        unsigned char last_char = examples[ex][last_step];
        size_t x_pos = (last_step * num_examples + ex) * onehot_dim + last_char;
        (*h_X)[x_pos] = 1.0f;
        
        // Target: we'll use a zero vector for simplicity (end of sequence marker)
        // This is already handled by calloc initialization
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int input_dim = 256;          // One-hot encoded byte
    int output_dim = 256;         // One-hot encoded output prediction
    int state_dim = 2048;         // State dimension
    float learning_rate = 0.0001; // Fixed learning rate
    int num_epochs = 300;         // Number of training epochs
    
    printf("=== SLM Training Configuration ===\n");
    printf("Input/Output dimension: %d (one-hot encoded)\n", input_dim);
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
    
    // Prepare training data with one-hot encoding
    float *h_X, *h_y;
    prepare_training_data(examples, num_examples, seq_length, &h_X, &h_y);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    size_t x_data_size = num_examples * seq_length * input_dim * sizeof(float);
    size_t y_data_size = num_examples * seq_length * output_dim * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_X, x_data_size));
    CHECK_CUDA(cudaMalloc(&d_y, y_data_size));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, x_data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, y_data_size, cudaMemcpyHostToDevice));
    
    // Initialize model (each example is processed in parallel)
    int batch_size = num_examples;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
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
            float* step_X_ptr = &d_X[step * batch_size * input_dim];
            float* step_y_ptr = &d_y[step * batch_size * output_dim];
            
            // Forward pass
            forward_pass(ssm, step_X_ptr);
            
            // Calculate cross-entropy loss
            float step_loss = calculate_cross_entropy_loss(ssm, step_y_ptr);
            epoch_loss += step_loss;
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, step_X_ptr);
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