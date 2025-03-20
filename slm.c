#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include "gmlp/gpu/gmlp.h"
#include "embeddings.h"

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
void backward_between_models(GMLP* first_model, GMLP* second_model, float* d_first_model_input) {
    // Copy gradients from second model's input to first model's output error
    const float alpha = 1.0f, beta = 0.0f;
    
    // Relay error from second model to first model
    checkCublasErrors(cublasSgemm(second_model->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               first_model->output_dim, first_model->batch_size, second_model->input_dim,
                               &alpha,
                               second_model->d_proj_in_weight, second_model->hidden_dim,
                               second_model->d_proj_in_grad, second_model->hidden_dim,
                               &beta,
                               first_model->d_error, first_model->output_dim),
                      "Error propagation between models");
    
    // Now do the backward pass for the first model
    backward_pass_gmlp(first_model, d_first_model_input);
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

// ---------------------------------------------------------------------
// Function: Calculate cross-entropy loss
// ---------------------------------------------------------------------
float calculate_cross_entropy_loss(GMLP* gmlp, float* targets) {
    const int batch_size = gmlp->batch_size;
    const int vocab_size = gmlp->output_dim;
    
    // Copy predictions and targets back to host for loss calculation
    float* h_predictions = (float*)malloc(batch_size * vocab_size * sizeof(float));
    float* h_targets = (float*)malloc(batch_size * vocab_size * sizeof(float));
    
    checkCudaErrors(cudaMemcpy(h_predictions, gmlp->d_predictions, 
                              batch_size * vocab_size * sizeof(float), 
                              cudaMemcpyDeviceToHost),
                   "Copy predictions to host for loss");
    
    checkCudaErrors(cudaMemcpy(h_targets, targets, 
                              batch_size * vocab_size * sizeof(float), 
                              cudaMemcpyDeviceToHost),
                   "Copy targets to host for loss");
    
    // Calculate cross-entropy loss
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        // Apply softmax and calculate loss for each sample
        float max_val = -INFINITY;
        for (int j = 0; j < vocab_size; j++) {
            if (h_predictions[i * vocab_size + j] > max_val) {
                max_val = h_predictions[i * vocab_size + j];
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            sum_exp += expf(h_predictions[i * vocab_size + j] - max_val);
        }
        
        float sample_loss = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            float softmax_output = expf(h_predictions[i * vocab_size + j] - max_val) / sum_exp;
            float target = h_targets[i * vocab_size + j];
            
            if (target > 0.0f) {  // Only add loss for the true class (one-hot encoding)
                sample_loss -= logf(fmaxf(softmax_output, 1e-10f));
            }
        }
        
        loss += sample_loss;
    }
    
    loss /= batch_size;
    
    free(h_predictions);
    free(h_targets);
    
    return loss;
}

// ---------------------------------------------------------------------
// Function: Setup gradients for cross-entropy loss
// ---------------------------------------------------------------------
void setup_cross_entropy_gradient(GMLP* gmlp, float* d_targets) {
    const int batch_size = gmlp->batch_size;
    const int vocab_size = gmlp->output_dim;
    
    int threads_per_block = 256;
    int num_blocks = (batch_size * vocab_size + threads_per_block - 1) / threads_per_block;
    
    // Custom kernel to compute softmax gradient (predictions - targets)
    error_computation_kernel<<<num_blocks, threads_per_block>>>(
        gmlp->d_predictions, d_targets, gmlp->d_error, batch_size * vocab_size);
    
    checkCudaErrors(cudaGetLastError(), "Error computation kernel launch");
}

int main(int argc, char *argv[]) {
    // Seed random number generator
    srand(time(NULL) ^ getpid());
    
    // Model parameters
    int embedding_dim = 128;
    int layer_dims[6] = {
        256,  // Layer 1
        512,  // Layer 2
        768,  // Layer 3
        512,  // Layer 4
        384,  // Layer 5
        256   // Layer 6 (output layer)
    };
    int hidden_dims[6] = {
        512,  // Layer 1
        768,  // Layer 2
        1024, // Layer 3
        768,  // Layer 4
        512,  // Layer 5
        384   // Layer 6
    };
    int ffn_dims[6] = {
        1024, // Layer 1
        1536, // Layer 2
        2048, // Layer 3
        1536, // Layer 4
        1024, // Layer 5
        768   // Layer 6
    };
    int vocab_size = 256;
    float learning_rate = 0.0005;
    int num_epochs = 50;
    int max_samples = 16384;
    float lr_min_ratio = 0.001f;
    
    printf("=== gMLP Language Model Training Configuration ===\n");
    printf("Vocabulary size: %d (byte values)\n", vocab_size);
    printf("Embedding dimension: %d\n", embedding_dim);
    
    for (int i = 0; i < 6; i++) {
        printf("Layer %d: hidden_dim=%d, ffn_dim=%d, output_dim=%d\n", 
               i+1, hidden_dims[i], ffn_dims[i], layer_dims[i]);
    }
    
    printf("Base learning rate: %.6f\n", learning_rate);
    printf("Min learning rate: %.6f\n", learning_rate * lr_min_ratio);
    printf("Learning rate schedule: Cosine decay\n");
    printf("Training epochs: %d\n", num_epochs);
    printf("Using first %d samples for training\n\n", max_samples);
    
    int batch_size = max_samples;
    int seq_length = 1024;
    int context_size = 128;  // Context window size for gMLP

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
    checkCudaErrors(cudaMalloc(&d_X_time_major, batch_size * seq_length * sizeof(unsigned char)),
                    "Allocate d_X_time_major");
    checkCudaErrors(cudaMalloc(&d_y_time_major, batch_size * seq_length * sizeof(unsigned char)),
                    "Allocate d_y_time_major");
    
    checkCudaErrors(cudaMemcpy(d_X_time_major, h_X_time_major, 
                              batch_size * seq_length * sizeof(unsigned char), 
                              cudaMemcpyHostToDevice),
                    "Copy X to device");
    checkCudaErrors(cudaMemcpy(d_y_time_major, h_y_time_major, 
                              batch_size * seq_length * sizeof(unsigned char), 
                              cudaMemcpyHostToDevice),
                    "Copy y to device");
    
    // Free host memory
    free(h_X_time_major);
    free(h_y_time_major);
    
    // Allocate memory for one-hot encoding for a single timestep
    float* d_y_onehot_t;
    checkCudaErrors(cudaMalloc(&d_y_onehot_t, batch_size * vocab_size * sizeof(float)),
                    "Allocate d_y_onehot_t");
    
    // Initialize models
    Embeddings* embeddings;
    GMLP* layers[6];  // Array of 6 layers
    
    // Load models or initialize from scratch
    if (argc == 8) {
        // Load embeddings and models from files
        char* embedding_filename = argv[7];
        embeddings = load_embeddings(embedding_filename);
        
        for (int i = 0; i < 6; i++) {
            // Custom load function for gMLP
            layers[i] = load_gmlp(argv[i+1], batch_size);
        }
        
        printf("Successfully loaded pretrained models\n");
    } else {
        // Initialize from scratch
        embeddings = init_embeddings(vocab_size, embedding_dim);
        
        // Initialize first layer with embedding dim as input
        layers[0] = init_gmlp(embedding_dim * context_size, hidden_dims[0], ffn_dims[0], layer_dims[0], batch_size);
        
        // Initialize internal layers
        for (int i = 1; i < 5; i++) {
            layers[i] = init_gmlp(layer_dims[i-1], hidden_dims[i], ffn_dims[i], layer_dims[i], batch_size);
        }
        
        // Initialize final layer (outputs to vocab_size)
        layers[5] = init_gmlp(layer_dims[4], hidden_dims[5], ffn_dims[5], vocab_size, batch_size);
        
        printf("Initialized new models\n");
    }
    
    // Calculate and print total parameter count
    long long total_params = 0;
    
    // Embedding parameters: vocab_size * embedding_dim
    long long embedding_params = (long long)vocab_size * embedding_dim;
    total_params += embedding_params;
    
    printf("Model parameter count:\n");
    printf("  Embeddings:  %lld parameters\n", embedding_params);
    
    // gMLP parameters for each layer
    for (int i = 0; i < 6; i++) {
        int input_dim = (i == 0) ? embedding_dim * context_size : layer_dims[i-1];
        int output_dim = (i == 5) ? vocab_size : layer_dims[i];
        int hidden_dim = hidden_dims[i];
        int ffn_dim = ffn_dims[i];
        int half_hidden = hidden_dim / 2;
        
        long long layer_params = (long long)hidden_dim * input_dim +                       // proj_in_weight
                                ffn_dim * half_hidden +                                    // sgu_gate_weight
                                ffn_dim * half_hidden +                                    // sgu_proj_weight
                                hidden_dim * ffn_dim +                                     // sgu_out_weight
                                output_dim * hidden_dim;                                   // proj_out_weight
        
        total_params += layer_params;
        printf("  Layer %d gMLP: %lld parameters\n", i+1, layer_params);
    }
    
    printf("  Total:       %lld parameters (%.2f million)\n", total_params, total_params / 1000000.0);

    // Allocate memory for intermediate outputs and context window
    float* d_X_embedded_window;
    float* d_layer_outputs[6];  // Outputs of each layer
    float* d_context_window;
    
    // Allocate memory for context window (batch_size x context_size x embedding_dim)
    checkCudaErrors(cudaMalloc(&d_context_window, batch_size * context_size * embedding_dim * sizeof(float)),
                    "Allocate d_context_window");
    
    // Allocate memory for flattened context window input to first layer
    checkCudaErrors(cudaMalloc(&d_X_embedded_window, batch_size * context_size * embedding_dim * sizeof(float)),
                    "Allocate d_X_embedded_window");
    
    for (int i = 0; i < 6; i++) {
        checkCudaErrors(cudaMalloc(&d_layer_outputs[i], batch_size * 
                                  ((i == 5) ? vocab_size : layer_dims[i]) * 
                                  sizeof(float)),
                        "Allocate d_layer_outputs");
    }
    
    // Allocate temporary GPU memory for a single embedding step
    float* d_X_embedded_t;
    checkCudaErrors(cudaMalloc(&d_X_embedded_t, batch_size * embedding_dim * sizeof(float)),
                    "Allocate d_X_embedded_t");
    
    // Configure kernel launch parameters for one-hot encoding
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("\nStarting training for %d epochs with %d samples...\n", num_epochs, batch_size);
    
    // Calculate total optimization steps for learning rate schedule
    int total_steps = num_epochs * (seq_length - context_size);
    int current_step = 0;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        // Initialize context window with zeros
        checkCudaErrors(cudaMemset(d_context_window, 0, batch_size * context_size * embedding_dim * sizeof(float)),
                       "Initialize d_context_window");
        
        // Process each timestep (need context_size context before we can make predictions)
        for (int t = 0; t < seq_length; t++) {
            // Get current timestep input
            unsigned char* d_X_t = d_X_time_major + t * batch_size;
            
            // Embed the current timestep
            embeddings_forward(embeddings, d_X_t, d_X_embedded_t, batch_size);
            
            // Shift context window to the left (discard oldest element)
            if (t >= context_size) {
                // Copy all but the first embedding in the window forward
                checkCudaErrors(cudaMemcpy(
                    d_context_window, 
                    d_context_window + batch_size * embedding_dim,
                    batch_size * (context_size - 1) * embedding_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice),
                    "Shift context window");
            }
            
            // Add the new embedding to the end of the context window
            checkCudaErrors(cudaMemcpy(
                d_context_window + batch_size * (context_size - 1) * embedding_dim,
                d_X_embedded_t,
                batch_size * embedding_dim * sizeof(float),
                cudaMemcpyDeviceToDevice),
                "Add new embedding to context window");
            
            // Skip prediction until we have enough context
            if (t < context_size - 1) continue;
            
            // Update current step and learning rate
            current_step = epoch * (seq_length - context_size) + (t - context_size + 1);
            float current_lr = get_cosine_lr(learning_rate, current_step, total_steps, lr_min_ratio);
            
            // Get target for current prediction
            unsigned char* d_y_t = d_y_time_major + t * batch_size;
            
            // Generate one-hot encoding for current timestep targets
            onehot_encode_timestep<<<num_blocks, threads_per_block>>>(
                d_y_t, d_y_onehot_t, batch_size, vocab_size);
            
            // Reshape context window from (batch_size, context_size, embedding_dim) to 
            // (batch_size, context_size * embedding_dim) for gMLP input
            // This is a linearized operation, just changing how we view the memory
            d_X_embedded_window = d_context_window;
            
            // Forward pass through all layers
            forward_pass_gmlp(layers[0], d_X_embedded_window);
            checkCudaErrors(cudaMemcpy(d_layer_outputs[0], layers[0]->d_predictions, 
                                      batch_size * layer_dims[0] * sizeof(float), 
                                      cudaMemcpyDeviceToDevice),
                           "Copy layer 0 output");
            
            for (int i = 1; i < 6; i++) {
                forward_pass_gmlp(layers[i], d_layer_outputs[i-1]);
                if (i < 5) {  // Don't need to copy for the last layer
                    checkCudaErrors(cudaMemcpy(d_layer_outputs[i], layers[i]->d_predictions, 
                                              batch_size * layer_dims[i] * sizeof(float), 
                                              cudaMemcpyDeviceToDevice),
                                   "Copy layer output");
                }
            }
            
            // Calculate loss
            float loss = calculate_cross_entropy_loss(layers[5], d_y_onehot_t);
            epoch_loss += loss;
            
            // Set up gradients for cross-entropy loss
            setup_cross_entropy_gradient(layers[5], d_y_onehot_t);
            
            // Backward pass through all layers
            backward_pass_gmlp(layers[5], d_layer_outputs[4]);
            
            for (int i = 4; i >= 0; i--) {
                float* previous_input = (i == 0) ? d_X_embedded_window : d_layer_outputs[i-1];
                backward_between_models(layers[i], layers[i+1], previous_input);
            }
            
            // Backward pass through embeddings (only for the newest token)
            // This is more complex with context window, need to accumulate gradients
            // We only backpropagate to the most recent token in the context window
            float* d_embedding_grad = (float*)malloc(batch_size * embedding_dim * sizeof(float));
            
            // Extract the gradient for the newest token from the context window gradient
            checkCudaErrors(cudaMemcpy(
                d_embedding_grad,
                layers[0]->d_error + batch_size * (context_size - 1) * embedding_dim,
                batch_size * embedding_dim * sizeof(float),
                cudaMemcpyDeviceToHost),
                "Extract embedding gradient");
            
            // Update embeddings
            embeddings_backward(embeddings, d_embedding_grad, d_X_t, batch_size);
            free(d_embedding_grad);
            
            // Update weights using current learning rate
            for (int i = 0; i < 6; i++) {
                update_weights_gmlp(layers[i], current_lr);
            }
            update_embeddings(embeddings, current_lr, batch_size);
            
            // Print progress
            if ((t - context_size + 1) % 100 == 0) {
                printf("Epoch %d/%d, Step %d/%d, LR: %.7f, Loss: %f\n", epoch + 1, 
                    num_epochs, t - context_size + 2, seq_length - context_size + 1, 
                    current_lr, loss);
            }
        }
        
        float avg_epoch_loss = epoch_loss / (seq_length - context_size + 1);
        printf("Epoch %d/%d completed, Average Loss: %f\n", epoch + 1, num_epochs, avg_epoch_loss);
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
    for (int i = 0; i < 6; i++) {
        char layer_fname[64];
        sprintf(layer_fname, "%s_layer%d.bin", model_time, i+1);
        save_gmlp(layers[i], layer_fname);
    }

    // Clean up
    for (int i = 0; i < 6; i++) {
        free_gmlp(layers[i]);
        cudaFree(d_layer_outputs[i]);
    }
    free_embeddings(embeddings);
    
    cudaFree(d_X_time_major);
    cudaFree(d_y_time_major);
    cudaFree(d_context_window);
    cudaFree(d_X_embedded_t);
    cudaFree(d_y_onehot_t);
    
    printf("\nTraining completed! Models saved with prefix: %s\n", model_time);
    return 0;
}