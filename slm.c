#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include "ssm/gpu/ssm.h"
#include "gpu/embeddings.h"

// ---------------------------------------------------------------------
// CUDA kernel for one-hot encoding a single timestep
// ---------------------------------------------------------------------
__global__ void onehot_encode_timestep(const unsigned char* input, float* output, 
                                     int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx < batch_size) {
        unsigned char target = input[batch_idx];
        
        // Clear all values first (set to zero)
        for (int v = 0; v < vocab_size; v++) {
            output[batch_idx * vocab_size + v] = 0.0f;
        }
        
        // Set the target index to 1.0
        output[batch_idx * vocab_size + target] = 1.0f;
    }
}

// ---------------------------------------------------------------------
// Function: Propagate gradients between stacked models
// ---------------------------------------------------------------------
void backward_between_models(SSM* first_model, SSM* second_model, float* d_first_model_input) {
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

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 24;
    int layer1_dim = 96;
    int layer2_dim = 160;
    int layer3_dim = 224;
    int layer4_dim = 320;
    int layer5_dim = 320;
    int layer6_dim = 224;
    int layer7_dim = 160;
    int layer8_dim = 320;
    int state_dim = 896;
    int vocab_size = 256;
    float learning_rate = 0.001;
    int num_epochs = 10;
    int max_samples = 131072;
    int grad_accum_steps = 8;
    
    printf("=== SLM Training Configuration ===\n");
    printf("Vocabulary size: %d (byte values)\n", vocab_size);
    printf("Embedding dimension: %d\n", embedding_dim);
    printf("Layer 1 dimension: %d\n", layer1_dim);
    printf("Layer 2 dimension: %d\n", layer2_dim);
    printf("Layer 3 dimension: %d\n", layer3_dim);
    printf("Layer 4 dimension: %d\n", layer4_dim);
    printf("Layer 5 dimension: %d\n", layer5_dim);
    printf("Layer 6 dimension: %d\n", layer6_dim);
    printf("Layer 7 dimension: %d\n", layer7_dim);
    printf("Layer 8 dimension: %d\n", layer8_dim);
    printf("State dimension: %d\n", state_dim);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("Training epochs: %d\n", num_epochs);
    printf("Using first %d samples for training\n", max_samples);
    printf("Gradient accumulation steps: %d\n\n", grad_accum_steps);
    
    int batch_size = max_samples;
    int seq_length = 1024;

    // Allocate memory for samples in time-major format
    unsigned char** samples = (unsigned char**)malloc(batch_size * sizeof(unsigned char*));
    for (int i = 0; i < batch_size; i++) {
        samples[i] = (unsigned char*)malloc(seq_length * sizeof(unsigned char));
    }
    
    // Read samples into memory
    FILE* file = fopen("data.txt", "rb");
    char* line_buf = NULL;
    size_t line_buf_size = 0;
    ssize_t line_len;
    int sample_idx = 0;
    
    while ((line_len = getline(&line_buf, &line_buf_size, file)) != -1 && sample_idx < batch_size) {
        memcpy(samples[sample_idx], line_buf, seq_length);
        sample_idx++;
    }
    
    free(line_buf);
    fclose(file);
    
    // Allocate time-major formatted input/target arrays
    unsigned char* h_X_time_major = (unsigned char*)malloc(batch_size * seq_length * sizeof(unsigned char));
    unsigned char* h_y_time_major = (unsigned char*)malloc(batch_size * seq_length * sizeof(unsigned char));
    
    // Fill arrays in time-major format
    for (int t = 0; t < seq_length; t++) {
        for (int b = 0; b < batch_size; b++) {
            int idx = t * batch_size + b;
            h_X_time_major[idx] = samples[b][t];
            // Next character or wrap to beginning
            h_y_time_major[idx] = (t < seq_length-1) ? samples[b][t+1] : samples[b][0];
        }
    }
    
    // Free sample memory
    for (int i = 0; i < batch_size; i++) {
        free(samples[i]);
    }
    free(samples);
    
    // Allocate GPU memory for input and target data
    unsigned char *d_X_time_major, *d_y_time_major;
    CHECK_CUDA(cudaMalloc(&d_X_time_major, batch_size * seq_length * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_y_time_major, batch_size * seq_length * sizeof(unsigned char)));
    
    CHECK_CUDA(cudaMemcpy(d_X_time_major, h_X_time_major, 
                         batch_size * seq_length * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_time_major, h_y_time_major, 
                         batch_size * seq_length * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    
    // Free host memory
    free(h_X_time_major);
    free(h_y_time_major);
    
    // Allocate memory for one-hot encoding for a single timestep
    float* d_y_onehot_t;
    CHECK_CUDA(cudaMalloc(&d_y_onehot_t, batch_size * vocab_size * sizeof(float)));
    
    // Initialize models
    Embeddings* embeddings;
    SSM* layer1_ssm;
    SSM* layer2_ssm;
    SSM* layer3_ssm;
    SSM* layer4_ssm;
    SSM* layer5_ssm;
    SSM* layer6_ssm;
    SSM* layer7_ssm;
    SSM* layer8_ssm;
    
    if (argc == 10) {
        // Load embeddings and models from files
        char* layer1_filename = argv[1];
        char* layer2_filename = argv[2];
        char* layer3_filename = argv[3];
        char* layer4_filename = argv[4];
        char* layer5_filename = argv[5];
        char* layer6_filename = argv[6];
        char* layer7_filename = argv[7];
        char* layer8_filename = argv[8];
        char* embedding_filename = argv[9];

        embeddings = load_embeddings(embedding_filename);
        layer1_ssm = load_ssm(layer1_filename, batch_size);
        layer2_ssm = load_ssm(layer2_filename, batch_size);
        layer3_ssm = load_ssm(layer3_filename, batch_size);
        layer4_ssm = load_ssm(layer4_filename, batch_size);
        layer5_ssm = load_ssm(layer5_filename, batch_size);
        layer6_ssm = load_ssm(layer6_filename, batch_size);
        layer7_ssm = load_ssm(layer7_filename, batch_size);
        layer8_ssm = load_ssm(layer8_filename, batch_size);
        
        printf("Successfully loaded pretrained models\n");
    } else {
        // Initialize from scratch
        embeddings = init_embeddings(vocab_size, embedding_dim);
        layer1_ssm = init_ssm(embedding_dim, state_dim, layer1_dim, batch_size);
        layer2_ssm = init_ssm(layer1_dim, state_dim, layer2_dim, batch_size);
        layer3_ssm = init_ssm(layer2_dim, state_dim, layer3_dim, batch_size);
        layer4_ssm = init_ssm(layer3_dim, state_dim, layer4_dim, batch_size);
        layer5_ssm = init_ssm(layer4_dim, state_dim, layer5_dim, batch_size);
        layer6_ssm = init_ssm(layer5_dim, state_dim, layer6_dim, batch_size);
        layer7_ssm = init_ssm(layer6_dim, state_dim, layer7_dim, batch_size);
        layer8_ssm = init_ssm(layer7_dim, state_dim, vocab_size, batch_size);
        
        printf("Initialized new models\n");
    }
    
    // Calculate and print total parameter count
    long long total_params = 0;
    
    // Embedding parameters: vocab_size * embedding_dim
    long long embedding_params = (long long)vocab_size * embedding_dim;
    
    // SSM parameters for each layer
    long long layer1_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * embedding_dim +         // B matrix
                              layer1_dim * state_dim +            // C matrix
                              layer1_dim * embedding_dim;         // D matrix
    
    long long layer2_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer1_dim +            // B matrix
                              layer2_dim * state_dim +            // C matrix
                              layer2_dim * layer1_dim;            // D matrix
    
    long long layer3_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer2_dim +            // B matrix
                              layer3_dim * state_dim +            // C matrix
                              layer3_dim * layer2_dim;            // D matrix
    
    long long layer4_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer3_dim +            // B matrix
                              layer4_dim * state_dim +            // C matrix
                              layer4_dim * layer3_dim;            // D matrix
    
    long long layer5_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer4_dim +            // B matrix
                              layer5_dim * state_dim +            // C matrix
                              layer5_dim * layer4_dim;            // D matrix
    
    long long layer6_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer5_dim +            // B matrix
                              layer6_dim * state_dim +            // C matrix
                              layer6_dim * layer5_dim;            // D matrix
    
    long long layer7_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer6_dim +            // B matrix
                              layer7_dim * state_dim +            // C matrix
                              layer7_dim * layer6_dim;            // D matrix
    
    long long layer8_params = (long long)state_dim * state_dim +  // A matrix
                              state_dim * layer7_dim +            // B matrix
                              vocab_size * state_dim +            // C matrix
                              vocab_size * layer7_dim;            // D matrix
    
    total_params = embedding_params + layer1_params + layer2_params + layer3_params + 
                  layer4_params + layer5_params + layer6_params + layer7_params + layer8_params;
    
    printf("Model parameter count:\n");
    printf("  Embeddings:  %lld parameters\n", embedding_params);
    printf("  Layer 1 SSM: %lld parameters\n", layer1_params);
    printf("  Layer 2 SSM: %lld parameters\n", layer2_params);
    printf("  Layer 3 SSM: %lld parameters\n", layer3_params);
    printf("  Layer 4 SSM: %lld parameters\n", layer4_params);
    printf("  Layer 5 SSM: %lld parameters\n", layer5_params);
    printf("  Layer 6 SSM: %lld parameters\n", layer6_params);
    printf("  Layer 7 SSM: %lld parameters\n", layer7_params);
    printf("  Layer 8 SSM: %lld parameters\n", layer8_params);
    printf("  Total:       %lld parameters (%.2f million)\n", total_params, total_params / 1000000.0);

    // Allocate memory for embedded inputs and intermediate outputs
    float* d_X_embedded;
    float* d_layer1_output;
    float* d_layer2_output;
    float* d_layer3_output;
    float* d_layer4_output;
    float* d_layer5_output;
    float* d_layer6_output;
    float* d_layer7_output;

    CHECK_CUDA(cudaMalloc(&d_X_embedded, batch_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer1_output, batch_size * layer1_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, batch_size * layer2_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, batch_size * layer3_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer4_output, batch_size * layer4_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer5_output, batch_size * layer5_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer6_output, batch_size * layer6_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer7_output, batch_size * layer7_dim * sizeof(float)));
    
    // Configure kernel launch parameters for one-hot encoding
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("\nStarting training for %d epochs with %d samples...\n", num_epochs, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer5_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer6_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer7_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(layer8_ssm->d_state, 0, batch_size * state_dim * sizeof(float)));
        
        float epoch_loss = 0.0f;
        
        // Process each timestep
        for (int t = 0; t < seq_length; t++) {
            // Get current timestep inputs
            unsigned char* d_X_t = d_X_time_major + t * batch_size;
            unsigned char* d_y_t = d_y_time_major + t * batch_size;
            
            // Generate one-hot encoding for current timestep targets
            onehot_encode_timestep<<<num_blocks, threads_per_block>>>(
                d_y_t, d_y_onehot_t, batch_size, vocab_size);
            
            // Forward pass: embedding layer
            embeddings_forward(embeddings, d_X_t, d_X_embedded, batch_size);
            
            // Forward pass: layer 1 SSM
            forward_pass(layer1_ssm, d_X_embedded);
            
            // Copy layer1 output for layer2 input
            CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                              batch_size * layer1_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 2 SSM
            forward_pass(layer2_ssm, d_layer1_output);
            
            // Copy layer2 output for layer3 input
            CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                              batch_size * layer2_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 3 SSM
            forward_pass(layer3_ssm, d_layer2_output);
            
            // Copy layer3 output for layer4 input
            CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                              batch_size * layer3_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 4 SSM
            forward_pass(layer4_ssm, d_layer3_output);
            
            // Copy layer4 output for layer5 input
            CHECK_CUDA(cudaMemcpy(d_layer4_output, layer4_ssm->d_predictions, 
                              batch_size * layer4_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 5 SSM
            forward_pass(layer5_ssm, d_layer4_output);
            
            // Copy layer5 output for layer6 input
            CHECK_CUDA(cudaMemcpy(d_layer5_output, layer5_ssm->d_predictions, 
                              batch_size * layer5_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 6 SSM
            forward_pass(layer6_ssm, d_layer5_output);
            
            // Copy layer6 output for layer7 input
            CHECK_CUDA(cudaMemcpy(d_layer6_output, layer6_ssm->d_predictions, 
                              batch_size * layer6_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 7 SSM
            forward_pass(layer7_ssm, d_layer6_output);
            
            // Copy layer7 output for layer8 input
            CHECK_CUDA(cudaMemcpy(d_layer7_output, layer7_ssm->d_predictions, 
                              batch_size * layer7_dim * sizeof(float), 
                              cudaMemcpyDeviceToDevice));
            
            // Forward pass: layer 8 SSM (output layer)
            forward_pass(layer8_ssm, d_layer7_output);
            
            // Calculate loss
            float loss = calculate_cross_entropy_loss(layer8_ssm, d_y_onehot_t);
            epoch_loss += loss;
            
            // Backward pass: layer 8 SSM (output layer)
            backward_pass(layer8_ssm, d_layer7_output);
            
            // Backward pass: layer 7 SSM
            backward_between_models(layer7_ssm, layer8_ssm, d_layer6_output);
            
            // Backward pass: layer 6 SSM
            backward_between_models(layer6_ssm, layer7_ssm, d_layer5_output);
            
            // Backward pass: layer 5 SSM
            backward_between_models(layer5_ssm, layer6_ssm, d_layer4_output);
            
            // Backward pass: layer 4 SSM
            backward_between_models(layer4_ssm, layer5_ssm, d_layer3_output);
            
            // Backward pass: layer 3 SSM
            backward_between_models(layer3_ssm, layer4_ssm, d_layer2_output);
            
            // Backward pass: layer 2 SSM
            backward_between_models(layer2_ssm, layer3_ssm, d_layer1_output);
            
            // Backward pass: layer 1 SSM
            backward_between_models(layer1_ssm, layer2_ssm, d_X_embedded);
            
            // Backward pass: embeddings
            embeddings_backward(embeddings, layer1_ssm->d_error, d_X_t, batch_size);
            
            // Update weights every grad_accum_steps or at the end of sequence
            if ((t + 1) % grad_accum_steps == 0 || t == seq_length - 1) {
                update_weights(layer1_ssm, learning_rate);
                update_weights(layer2_ssm, learning_rate);
                update_weights(layer3_ssm, learning_rate);
                update_weights(layer4_ssm, learning_rate);
                update_weights(layer5_ssm, learning_rate);
                update_weights(layer6_ssm, learning_rate);
                update_weights(layer7_ssm, learning_rate);
                update_weights(layer8_ssm, learning_rate);
                update_embeddings(embeddings, learning_rate, batch_size);
                
                // Zero gradients after update
                zero_gradients(layer1_ssm);
                zero_gradients(layer2_ssm);
                zero_gradients(layer3_ssm);
                zero_gradients(layer4_ssm);
                zero_gradients(layer5_ssm);
                zero_gradients(layer6_ssm);
                zero_gradients(layer7_ssm);
                zero_gradients(layer8_ssm);
                zero_embedding_gradients(embeddings);
            }

            // Print progress
            if (t == 0 || t == seq_length - 1 || (t + 1) % 20 == 0) {
                printf("Epoch %d/%d, Step %d/%d, Average Loss: %f\n", epoch + 1, 
                    num_epochs, t + 1, seq_length, epoch_loss/(t+1));
            }
        }
    }
    
    // Save the final models
    char model_time[20];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    strftime(model_time, sizeof(model_time), "%Y%m%d_%H%M%S", timeinfo);
    
    char layer1_fname[64], layer2_fname[64], layer3_fname[64], layer4_fname[64];
    char layer5_fname[64], layer6_fname[64], layer7_fname[64], layer8_fname[64];
    char embedding_fname[64];
    
    sprintf(layer1_fname, "%s_layer1.bin", model_time);
    sprintf(layer2_fname, "%s_layer2.bin", model_time);
    sprintf(layer3_fname, "%s_layer3.bin", model_time);
    sprintf(layer4_fname, "%s_layer4.bin", model_time);
    sprintf(layer5_fname, "%s_layer5.bin", model_time);
    sprintf(layer6_fname, "%s_layer6.bin", model_time);
    sprintf(layer7_fname, "%s_layer7.bin", model_time);
    sprintf(layer8_fname, "%s_layer8.bin", model_time);
    sprintf(embedding_fname, "%s_embeddings.bin", model_time);
    
    save_ssm(layer1_ssm, layer1_fname);
    save_ssm(layer2_ssm, layer2_fname);
    save_ssm(layer3_ssm, layer3_fname);
    save_ssm(layer4_ssm, layer4_fname);
    save_ssm(layer5_ssm, layer5_fname);
    save_ssm(layer6_ssm, layer6_fname);
    save_ssm(layer7_ssm, layer7_fname);
    save_ssm(layer8_ssm, layer8_fname);
    save_embeddings(embeddings, embedding_fname);

    // Clean up
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    free_ssm(layer5_ssm);
    free_ssm(layer6_ssm);
    free_ssm(layer7_ssm);
    free_ssm(layer8_ssm);
    free_embeddings(embeddings);
    
    cudaFree(d_X_time_major);
    cudaFree(d_y_time_major);
    cudaFree(d_X_embedded);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    cudaFree(d_layer4_output);
    cudaFree(d_layer5_output);
    cudaFree(d_layer6_output);
    cudaFree(d_layer7_output);
    cudaFree(d_y_onehot_t);
    
    printf("\nTraining completed!\n");
    return 0;
}