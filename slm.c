#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"
#include "embeddings.h"

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

// ---------------------------------------------------------------------
// Function: Propagate gradients between stacked models
// ---------------------------------------------------------------------
void backward_between_models(SSM* first_model, SSM* second_model, float* d_first_model_input) {
    // Zero gradients for first model
    zero_gradients(first_model);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Compute gradient from state path: d_input_grad = B^T * state_error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->state_dim,
                           &alpha,
                           second_model->d_B, second_model->state_dim,
                           second_model->d_state_error, second_model->state_dim,
                           &beta,
                           first_model->d_error, first_model->output_dim));
    
    // Add gradient from direct path: d_input_grad += D^T * error
    CHECK_CUBLAS(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           first_model->output_dim, first_model->batch_size, second_model->output_dim,
                           &alpha,
                           second_model->d_D, second_model->output_dim,
                           second_model->d_error, second_model->output_dim,
                           &alpha, // Add to existing gradient
                           first_model->d_error, first_model->output_dim));
    
    // Now do the backward pass for the first model
    backward_pass(first_model, d_first_model_input);
}

int main() {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 8;     // Embedding dimension
    int encoding_dim = 256;      // Encoding layer dimension
    int reasoning_dim = 128;     // Reasoning layer dimension
    int state_dim = 512;        // State dimension
    int vocab_size = 256;        // One per possible byte value
    float learning_rate = 0.00001; // Learning rate
    int num_epochs = 20;        // Number of training epochs
    int max_samples = 16384;       // Maximum number of samples to use
    
    printf("=== SLM Training Configuration ===\n");
    printf("Vocabulary size: %d (byte values)\n", vocab_size);
    printf("Embedding dimension: %d\n", embedding_dim);
    printf("Encoding dimension: %d\n", encoding_dim);
    printf("Reasoning dimension: %d\n", reasoning_dim);
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("Training epochs: %d\n", num_epochs);
    printf("Using first %d samples for training\n\n", max_samples);
    
    // Read training data
    FILE* file = fopen("data.txt", "rb");
    if (!file) {
        fprintf(stderr, "Error opening data.txt\n");
        return EXIT_FAILURE;
    }
    
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
    
    // Pre-scan to count non-empty lines (training samples) up to max_samples
    int num_training_samples = 0;
    char* scan_ptr = text_data;
    while (*scan_ptr && num_training_samples < max_samples) {
        // Skip to the next line start
        char* line_start = scan_ptr;
        
        // Find the end of the current line
        while (*scan_ptr && *scan_ptr != '\n') {
            scan_ptr++;
        }
        
        // Check if the line has content (non-empty)
        if (scan_ptr > line_start) {
            num_training_samples++;
        }
        
        // Move past the newline if present
        if (*scan_ptr == '\n') {
            scan_ptr++;
        }
    }
    
    printf("Found %d training samples (limited to first %d)\n", num_training_samples, max_samples);
    
    // Use the first max_samples samples for batch processing
    int batch_size = num_training_samples;
    printf("Using batch size: %d (processing samples in parallel)\n", batch_size);
    
    // Split data into training samples (each line is an example)
    char** training_samples = (char**)malloc(batch_size * sizeof(char*));
    
    // Load all available training samples (single pass)
    int sample_count = 0;
    char* line = strtok(text_data, "\n");
    while (line && sample_count < batch_size) {
        size_t len = strlen(line);
        if (len > 0) {  // Skip empty lines
            training_samples[sample_count] = (char*)malloc(len + 1);
            strcpy(training_samples[sample_count], line);
            sample_count++;
        }
        line = strtok(NULL, "\n");
    }
    
    printf("Loaded %d samples for training\n", sample_count);
    
    // Find the minimum length among training samples
    int min_length = strlen(training_samples[0]);
    int min_length_idx = 0;
    for (int i = 1; i < sample_count; i++) {
        int len = strlen(training_samples[i]);
        if (len < min_length) {
            min_length = len;
            min_length_idx = i;
        }
    }

    printf("Shortest sample is line %d with length %d\n", min_length_idx + 1, min_length);

    // Use sequence length of minimum sample length
    int seq_length = min_length;
    printf("Using sequence length: %d (from shortest sample)\n", seq_length);
    
    // Prepare input and target data
    unsigned char* h_X_data = (unsigned char*)malloc(sample_count * seq_length * sizeof(unsigned char));
    unsigned char* h_y_data = (unsigned char*)malloc(sample_count * seq_length * sizeof(unsigned char));
    
    // Copy data and set up X (current char) and y (next char)
    for (int ex = 0; ex < sample_count; ex++) {
        for (int pos = 0; pos < seq_length - 1; pos++) {
            h_X_data[ex * seq_length + pos] = training_samples[ex][pos];
            h_y_data[ex * seq_length + pos] = training_samples[ex][pos + 1];
        }
        // For the last position, use a wrap-around (last char predicts first char)
        h_X_data[ex * seq_length + (seq_length - 1)] = training_samples[ex][seq_length - 1];
        h_y_data[ex * seq_length + (seq_length - 1)] = training_samples[ex][0];
    }
    
    // Reorganize data to [time][example] layout for efficient processing
    unsigned char* h_X_time_major = (unsigned char*)malloc(sample_count * seq_length * sizeof(unsigned char));
    unsigned char* h_y_time_major = (unsigned char*)malloc(sample_count * seq_length * sizeof(unsigned char));
    
    reorganize_data(h_X_data, h_X_time_major, sample_count, seq_length);
    reorganize_data(h_y_data, h_y_time_major, sample_count, seq_length);
    
    // Free original arrays
    free(h_X_data);
    free(h_y_data);
    
    // Create one-hot encoded targets
    float* h_y_onehot = (float*)calloc(sample_count * seq_length * vocab_size, sizeof(float));
    for (int t = 0; t < seq_length; t++) {
        for (int ex = 0; ex < sample_count; ex++) {
            int idx = t * sample_count + ex;
            unsigned char target = h_y_time_major[idx];
            h_y_onehot[idx * vocab_size + target] = 1.0f;
        }
    }
    
    // Transfer data to GPU
    unsigned char* d_X_bytes;
    float* d_y_onehot;
    
    CHECK_CUDA(cudaMalloc(&d_X_bytes, sample_count * seq_length * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_y_onehot, sample_count * seq_length * vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_X_bytes, h_X_time_major, 
                         sample_count * seq_length * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_onehot, h_y_onehot, 
                         sample_count * seq_length * vocab_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Initialize the embedding layer
    Embeddings* embeddings = init_embeddings(vocab_size, embedding_dim);
    
    // Allocate memory for embedded inputs and intermediate outputs
    float* d_X_embedded;
    float* d_encoding_output;
    float* d_reasoning_output;
    CHECK_CUDA(cudaMalloc(&d_X_embedded, sample_count * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_encoding_output, sample_count * encoding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_reasoning_output, sample_count * reasoning_dim * sizeof(float)));
    
    // Initialize the stacked SSM models
    // 1. Encoder model processes embeddings -> encoding_dim
    // 2. Reasoning model processes encoding_dim -> reasoning_dim
    // 3. Output model processes reasoning_dim -> vocab_size
    SSM* encoder_ssm = init_ssm(embedding_dim, state_dim, encoding_dim, sample_count);
    SSM* reasoning_ssm = init_ssm(encoding_dim, state_dim, reasoning_dim, sample_count);
    SSM* output_ssm = init_ssm(reasoning_dim, state_dim, vocab_size, sample_count);
    
    printf("\nStarting training for %d epochs with all %d samples...\n", num_epochs, sample_count);
    printf("Using three-stage stacked SSM architecture\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(encoder_ssm->d_state, 0, sample_count * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(reasoning_ssm->d_state, 0, sample_count * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(output_ssm->d_state, 0, sample_count * state_dim * sizeof(float)));
        
        float epoch_loss = 0.0f;
        
        // Process each timestep
        for (int t = 0; t < seq_length; t++) {
            // Get current timestep data
            unsigned char* d_X_t = d_X_bytes + t * sample_count;
            float* d_y_t = d_y_onehot + t * sample_count * vocab_size;
            
            // Forward pass: embedding layer
            embeddings_forward(embeddings, d_X_t, d_X_embedded, sample_count);
            
            // Forward pass: encoder SSM layer
            forward_pass(encoder_ssm, d_X_embedded);
            
            // Copy encoder output for reasoning model input
            CHECK_CUDA(cudaMemcpy(d_encoding_output, encoder_ssm->d_predictions, 
                              sample_count * encoding_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: reasoning SSM layer
            forward_pass(reasoning_ssm, d_encoding_output);
            
            // Copy reasoning output for output model input
            CHECK_CUDA(cudaMemcpy(d_reasoning_output, reasoning_ssm->d_predictions, 
                              sample_count * reasoning_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: output SSM layer
            forward_pass(output_ssm, d_reasoning_output);
            
            // Calculate loss
            float loss = calculate_cross_entropy_loss(output_ssm, d_y_t);
            epoch_loss += loss;
            
            // Backward pass: output SSM layer
            zero_gradients(output_ssm);
            backward_pass(output_ssm, d_reasoning_output);
            
            // Backward pass: reasoning SSM layer
            backward_between_models(reasoning_ssm, output_ssm, d_encoding_output);
            
            // Backward pass: encoder SSM layer
            backward_between_models(encoder_ssm, reasoning_ssm, d_X_embedded);
            
            // Backward pass: embeddings
            zero_embedding_gradients(embeddings);
            embeddings_backward(embeddings, encoder_ssm->d_B_grad, d_X_t, sample_count);
            
            // Update weights
            update_weights(encoder_ssm, learning_rate);
            update_weights(reasoning_ssm, learning_rate);
            update_weights(output_ssm, learning_rate);
            update_embeddings(embeddings, learning_rate, sample_count);
        }
        
        // Calculate average loss
        float avg_loss = epoch_loss / seq_length;
        
        // Print progress every epoch, but always print first and last
        if (epoch == 0 || epoch == num_epochs - 1 || (epoch + 1) % 1 == 0) {
            printf("Epoch %d/%d, Average Loss: %.6f\n", 
                   epoch + 1, num_epochs, avg_loss);
        }
    }
    
    // Save the final models
    char model_time[20];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    strftime(model_time, sizeof(model_time), "%Y%m%d_%H%M%S", timeinfo);
    
    char encoder_fname[64], reasoning_fname[64], output_fname[64], embedding_fname[64];
    sprintf(encoder_fname, "%s_encoder.bin", model_time);
    sprintf(reasoning_fname, "%s_reasoning.bin", model_time);
    sprintf(output_fname, "%s_output.bin", model_time);
    sprintf(embedding_fname, "%s_embeddings.bin", model_time);
    
    // For inference, we need batch_size=1
    encoder_ssm->batch_size = 1;
    reasoning_ssm->batch_size = 1;
    output_ssm->batch_size = 1;
    
    save_ssm(encoder_ssm, encoder_fname);
    save_ssm(reasoning_ssm, reasoning_fname);
    save_ssm(output_ssm, output_fname);
    save_embeddings(embeddings, embedding_fname);

    // Clean up
    free_ssm(encoder_ssm);
    free_ssm(reasoning_ssm);
    free_ssm(output_ssm);
    free_embeddings(embeddings);
    
    for (int i = 0; i < sample_count; i++) {
        free(training_samples[i]);
    }
    free(training_samples);
    free(text_data);
    free(h_X_time_major);
    free(h_y_time_major);
    free(h_y_onehot);
    
    cudaFree(d_X_bytes);
    cudaFree(d_X_embedded);
    cudaFree(d_encoding_output);
    cudaFree(d_reasoning_output);
    cudaFree(d_y_onehot);
    
    printf("\nTraining completed!\n");
    return 0;
}