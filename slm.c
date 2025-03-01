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

// Split text into sequences of specified length
void prepare_training_data(const char* text, size_t text_length, 
                          int seq_length, int* num_sequences,
                          float** h_X, float** h_y) {
    // Calculate number of complete sequences we can extract
    *num_sequences = (text_length - 1) / seq_length;
    printf("Creating %d sequences of length %d\n", *num_sequences, seq_length);
    
    // Allocate memory for input and target sequences
    size_t data_size = (*num_sequences) * seq_length;
    *h_X = (float*)malloc(data_size * sizeof(float));
    *h_y = (float*)malloc(data_size * sizeof(float));
    
    if (!*h_X || !*h_y) {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(EXIT_FAILURE);
    }
    
    // Fill data arrays
    // For each sequence
    for (int seq = 0; seq < *num_sequences; seq++) {
        // For each position in the sequence
        for (int pos = 0; pos < seq_length; pos++) {
            int text_idx = seq * seq_length + pos;
            int data_idx = seq + pos * (*num_sequences); // Transpose for batch processing
            
            // Input: current character (normalized)
            (*h_X)[data_idx] = (unsigned char)text[text_idx] / 255.0f;
            
            // Target: next character (normalized)
            (*h_y)[data_idx] = (unsigned char)text[text_idx + 1] / 255.0f;
        }
    }
}

// Generate text using the trained model
void generate_text(SSM* ssm, const char* seed, int gen_length) {
    int seed_length = strlen(seed);
    char* generated = (char*)malloc(seed_length + gen_length + 1);
    if (!generated) {
        fprintf(stderr, "Memory allocation failed for text generation\n");
        return;
    }
    
    // Copy seed text
    strcpy(generated, seed);
    
    // Reset model state
    CHECK_CUDA(cudaMemset(ssm->d_state, 0, ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    // Prepare input
    float* h_input = (float*)malloc(sizeof(float));
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float)));
    
    // Process seed text through the model
    for (int i = 0; i < seed_length; i++) {
        h_input[0] = (unsigned char)seed[i] / 255.0f;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float), cudaMemcpyHostToDevice));
        forward_pass(ssm, d_input);
    }
    
    // Generate new text
    for (int i = 0; i < gen_length; i++) {
        // Get prediction
        float h_prediction;
        CHECK_CUDA(cudaMemcpy(&h_prediction, ssm->d_predictions, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Convert prediction to character
        h_prediction = fmaxf(0.0f, fminf(1.0f, h_prediction)) * 255.0f;
        unsigned char next_char = (unsigned char)roundf(h_prediction);
        
        // Add to generated text
        generated[seed_length + i] = next_char;
        
        // Feed back into model
        h_input[0] = next_char / 255.0f;
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float), cudaMemcpyHostToDevice));
        forward_pass(ssm, d_input);
    }
    
    // Null-terminate
    generated[seed_length + gen_length] = '\0';
    
    printf("\nGenerated text:\n%s\n", generated);
    
    free(generated);
    free(h_input);
    cudaFree(d_input);
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int input_dim = 1;      // Single byte input
    int output_dim = 1;     // Single byte output
    int state_dim = 512;    // Hidden state dimension
    int seq_length = 128;   // Sequence length for training
    float learning_rate = 0.0001f;
    int num_epochs = 50;
    
    // Read training data
    size_t text_length;
    char* text = read_text_file("data.txt", &text_length);
    printf("Loaded %zu characters of text data\n", text_length);
    
    // Prepare training data
    int num_sequences;
    float *h_X, *h_y;
    prepare_training_data(text, text_length, seq_length, &num_sequences, &h_X, &h_y);
    
    // Transfer data to GPU
    float *d_X, *d_y;
    size_t data_size = num_sequences * seq_length * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_X, data_size));
    CHECK_CUDA(cudaMalloc(&d_y, data_size));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, data_size, cudaMemcpyHostToDevice));
    
    // Initialize model (each sequence is processed in parallel)
    int batch_size = num_sequences;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Allocate memory for step data (one timestep across all sequences)
    float *d_step_X, *d_step_y;
    CHECK_CUDA(cudaMalloc(&d_step_X, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_step_y, batch_size * sizeof(float)));
    
    printf("Starting training for %d epochs (batch size: %d sequences)\n", 
           num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // Reset states at the beginning of each epoch
        CHECK_CUDA(cudaMemset(ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        // Process each timestep across all sequences
        for (int step = 0; step < seq_length; step++) {
            // Get step data (one character from each sequence)
            float* step_X_ptr = &d_X[step * batch_size];
            float* step_y_ptr = &d_y[step * batch_size];
            
            // Copy data to step buffers
            CHECK_CUDA(cudaMemcpy(d_step_X, step_X_ptr, batch_size * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_step_y, step_y_ptr, batch_size * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Forward pass
            forward_pass(ssm, d_step_X);
            
            // Calculate loss
            float step_loss = calculate_loss(ssm, d_step_y);
            epoch_loss += step_loss;
            
            // Backward pass and update
            zero_gradients(ssm);
            backward_pass(ssm, d_step_X);
            update_weights(ssm, learning_rate);
            
            // Print progress periodically
            if (step % 20 == 0) {
                printf("\rEpoch %d/%d, Step %d/%d, Loss: %.6f", 
                       epoch + 1, num_epochs, step + 1, seq_length, step_loss);
                fflush(stdout);
            }
        }
        
        // Print epoch statistics
        epoch_loss /= seq_length;
        printf("\nEpoch %d/%d completed, Average Loss: %.6f\n", 
               epoch + 1, num_epochs, epoch_loss);
        
        // Generate sample text periodically
        if ((epoch + 1) % 5 == 0 || epoch == num_epochs - 1) {
            // Create inference model with batch_size=1
            SSM* inference_ssm = init_ssm(input_dim, state_dim, output_dim, 1);
            
            // Copy trained weights
            CHECK_CUDA(cudaMemcpy(inference_ssm->d_A, ssm->d_A, 
                                 state_dim * state_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_ssm->d_B, ssm->d_B, 
                                 state_dim * input_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_ssm->d_C, ssm->d_C, 
                                 output_dim * state_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(inference_ssm->d_D, ssm->d_D, 
                                 output_dim * input_dim * sizeof(float), 
                                 cudaMemcpyDeviceToDevice));
            
            // Generate text
            const char* seed = "<USER> ";
            generate_text(inference_ssm, seed, 256);
            
            free_ssm(inference_ssm);
        }
    }
    
    printf("Training completed!\n");
    
    // Save model (create inference version with batch_size=1)
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
    
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_ssm(inference_model, model_fname);
    printf("Model saved to %s\n", model_fname);
    
    // Clean up
    free(text);
    free(h_X);
    free(h_y);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_step_X);
    cudaFree(d_step_y);
    free_ssm(ssm);
    free_ssm(inference_model);
    
    return 0;
}