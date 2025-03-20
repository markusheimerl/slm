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
    // Zero gradients for first model
    zero_gradients_ssm(first_model);
    
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
    backward_pass_ssm(first_model, d_first_model_input);
}

// ---------------------------------------------------------------------
// Function: Get learning rate with cosine schedule
// ---------------------------------------------------------------------
float get_cosine_lr(float base_lr, int current_step, int total_steps, float min_lr_ratio) {
    if (current_step >= total_steps) {
        return base_lr * min_lr_ratio;
    }
    
    float progress = (float)current_step / (float)total_steps;
    float cosine_decay = 0.5f * (1.0f + cosf(M_PI * progress));
    float decayed_lr = base_lr * ((1.0f - min_lr_ratio) * cosine_decay + min_lr_ratio);
    
    return decayed_lr;
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 24;
    int layer_dims[24] = {
        96,  // Layer 1
        128, // Layer 2
        160, // Layer 3
        192, // Layer 4
        224, // Layer 5
        256, // Layer 6
        288, // Layer 7
        320, // Layer 8
        320, // Layer 9
        352, // Layer 10
        352, // Layer 11
        384, // Layer 12
        384, // Layer 13
        384, // Layer 14
        352, // Layer 15
        352, // Layer 16
        320, // Layer 17
        320, // Layer 18
        288, // Layer 19
        256, // Layer 20
        224, // Layer 21
        192, // Layer 22
        160, // Layer 23
        256  // Layer 24 (output layer)
    };
    int state_dim = 896;
    int vocab_size = 256;
    float learning_rate = 0.0001;
    int num_epochs = 100;
    int max_samples = 32768;
    float lr_min_ratio = 0.001f;
    
    printf("=== SLM Training Configuration ===\n");
    printf("Vocabulary size: %d (byte values)\n", vocab_size);
    printf("Embedding dimension: %d\n", embedding_dim);
    
    for (int i = 0; i < 24; i++) {
        printf("Layer %d dimension: %d\n", i+1, layer_dims[i]);
    }
    
    printf("State dimension: %d\n", state_dim);
    printf("Base learning rate: %.6f\n", learning_rate);
    printf("Min learning rate: %.6f\n", learning_rate * lr_min_ratio);
    printf("Learning rate schedule: Cosine decay\n");
    printf("Training epochs: %d\n", num_epochs);
    printf("Using first %d samples for training\n\n", max_samples);
    
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
    SSM* layers[24];  // Array of 24 layers
    
    if (argc == 26) {
        // Load embeddings and models from files
        char* embedding_filename = argv[25];
        embeddings = load_embeddings(embedding_filename);
        
        for (int i = 0; i < 24; i++) {
            layers[i] = load_ssm(argv[i+1], batch_size);
        }
        
        printf("Successfully loaded pretrained models\n");
    } else {
        // Initialize from scratch
        embeddings = init_embeddings(vocab_size, embedding_dim);
        
        // Initialize first layer with embedding dim as input
        layers[0] = init_ssm(embedding_dim, state_dim, layer_dims[0], batch_size);
        
        // Initialize internal layers
        for (int i = 1; i < 23; i++) {
            layers[i] = init_ssm(layer_dims[i-1], state_dim, layer_dims[i], batch_size);
        }
        
        // Initialize final layer (outputs to vocab_size)
        layers[23] = init_ssm(layer_dims[22], state_dim, vocab_size, batch_size);
        
        printf("Initialized new models\n");
    }
    
    // Calculate and print total parameter count
    long long total_params = 0;
    
    // Embedding parameters: vocab_size * embedding_dim
    long long embedding_params = (long long)vocab_size * embedding_dim;
    total_params += embedding_params;
    
    printf("Model parameter count:\n");
    printf("  Embeddings:  %lld parameters\n", embedding_params);
    
    // SSM parameters for each layer
    for (int i = 0; i < 24; i++) {
        int input_dim = (i == 0) ? embedding_dim : layer_dims[i-1];
        int output_dim = (i == 23) ? vocab_size : layer_dims[i];
        
        long long layer_params = (long long)state_dim * state_dim +  // A matrix
                                state_dim * input_dim +              // B matrix
                                output_dim * state_dim +             // C matrix
                                output_dim * input_dim;              // D matrix
        
        total_params += layer_params;
        printf("  Layer %2d SSM: %lld parameters\n", i+1, layer_params);
    }
    
    printf("  Total:       %lld parameters (%.2f million)\n", total_params, total_params / 1000000.0);

    // Allocate memory for intermediate outputs
    float* d_X_embedded;
    float* d_layer_outputs[24];  // Outputs of each layer

    CHECK_CUDA(cudaMalloc(&d_X_embedded, batch_size * embedding_dim * sizeof(float)));
    
    for (int i = 0; i < 24; i++) {
        CHECK_CUDA(cudaMalloc(&d_layer_outputs[i], batch_size * 
                             ((i == 23) ? vocab_size : layer_dims[i]) * 
                             sizeof(float)));
    }
    
    // Configure kernel launch parameters for one-hot encoding
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("\nStarting training for %d epochs with %d samples...\n", num_epochs, batch_size);
    
    // Calculate total optimization steps for learning rate schedule
    int total_steps = num_epochs * seq_length;
    int current_step = 0;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Reset state at the beginning of each epoch
        for (int i = 0; i < 24; i++) {
            CHECK_CUDA(cudaMemset(layers[i]->d_state, 0, batch_size * state_dim * sizeof(float)));
        }
        
        float epoch_loss = 0.0f;
        
        // Process each timestep
        for (int t = 0; t < seq_length; t++) {
            // Update current step and learning rate
            current_step = epoch * seq_length + t;
            float current_lr = get_cosine_lr(learning_rate, current_step, total_steps, lr_min_ratio);
            
            // Get current timestep inputs
            unsigned char* d_X_t = d_X_time_major + t * batch_size;
            unsigned char* d_y_t = d_y_time_major + t * batch_size;
            
            // Generate one-hot encoding for current timestep targets
            onehot_encode_timestep<<<num_blocks, threads_per_block>>>(
                d_y_t, d_y_onehot_t, batch_size, vocab_size);
            
            // Forward pass: embedding layer
            embeddings_forward(embeddings, d_X_t, d_X_embedded, batch_size);
            
            // Forward pass through all 24 layers
            forward_pass_ssm(layers[0], d_X_embedded);
            CHECK_CUDA(cudaMemcpy(d_layer_outputs[0], layers[0]->d_predictions, 
                               batch_size * layer_dims[0] * sizeof(float), 
                               cudaMemcpyDeviceToDevice));
            
            for (int i = 1; i < 24; i++) {
                forward_pass_ssm(layers[i], d_layer_outputs[i-1]);
                if (i < 23) {  // Don't need to copy for the last layer
                    CHECK_CUDA(cudaMemcpy(d_layer_outputs[i], layers[i]->d_predictions, 
                                       batch_size * layer_dims[i] * sizeof(float), 
                                       cudaMemcpyDeviceToDevice));
                }
            }
            
            // Calculate loss
            float loss = calculate_cross_entropy_loss(layers[23], d_y_onehot_t);
            epoch_loss += loss;
            
            // Backward pass through all 24 layers
            zero_gradients_ssm(layers[23]);
            backward_pass_ssm(layers[23], d_layer_outputs[22]);
            
            for (int i = 22; i >= 0; i--) {
                float* previous_input = (i == 0) ? d_X_embedded : d_layer_outputs[i-1];
                backward_between_models(layers[i], layers[i+1], previous_input);
            }
            
            // Backward pass: embeddings
            zero_embedding_gradients(embeddings);
            embeddings_backward(embeddings, layers[0]->d_error, d_X_t, batch_size);
            
            // Update weights using current learning rate
            for (int i = 0; i < 24; i++) {
                update_weights_ssm(layers[i], current_lr);
            }
            update_embeddings(embeddings, current_lr, batch_size);            

            // Print progress
            if (t == 0 || t == seq_length - 1 || (t + 1) % 20 == 0) {
                printf("Epoch %d/%d, Step %d/%d, LR: %.7f, Average Loss: %f\n", epoch + 1, 
                    num_epochs, t + 1, seq_length, current_lr, epoch_loss/(t+1));
            }
        }
    }
    
    // Save the final models
    char model_time[20];
    time_t now = time(NULL);
    struct tm *timeinfo = localtime(&now);
    strftime(model_time, sizeof(model_time), "%Y%m%d_%H%M%S", timeinfo);
    
    char embedding_fname[64];
    sprintf(embedding_fname, "%s_embeddings.bin", model_time);
    save_embeddings(embeddings, embedding_fname);
    
    // Save all layer models
    for (int i = 0; i < 24; i++) {
        char layer_fname[64];
        sprintf(layer_fname, "%s_layer%d.bin", model_time, i+1);
        save_ssm(layers[i], layer_fname);
    }

    // Clean up
    for (int i = 0; i < 24; i++) {
        free_ssm(layers[i]);
        cudaFree(d_layer_outputs[i]);
    }
    free_embeddings(embeddings);
    
    cudaFree(d_X_time_major);
    cudaFree(d_y_time_major);
    cudaFree(d_X_embedded);
    cudaFree(d_y_onehot_t);
    
    printf("\nTraining completed!\n");
    return 0;
}