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