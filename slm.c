#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_data_for_batch_processing(float* X, float* y, 
                                     float** X_reshaped, float** y_reshaped,
                                     int num_sequences, int seq_len, 
                                     int input_dim, int output_dim) {
    // Reshape to: seq_len tensors of size (batch_size x input_dim/output_dim)
    *X_reshaped = (float*)malloc(seq_len * num_sequences * input_dim * sizeof(float));
    *y_reshaped = (float*)malloc(seq_len * num_sequences * output_dim * sizeof(float));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time][feature]
            int orig_x_idx = b * seq_len * input_dim + t * input_dim;
            int orig_y_idx = b * seq_len * output_dim + t * output_dim;
            
            // New layout: [time][seq][feature] 
            int new_x_idx = t * num_sequences * input_dim + b * input_dim;
            int new_y_idx = t * num_sequences * output_dim + b * output_dim;
            
            memcpy(&(*X_reshaped)[new_x_idx], &X[orig_x_idx], input_dim * sizeof(float));
            memcpy(&(*y_reshaped)[new_y_idx], &y[orig_y_idx], output_dim * sizeof(float));
        }
    }
}

int main() {
    srand(time(NULL));
    
    // Parameters
    const int input_dim = 512;      // EMBED_DIM - embedding dimension
    const int state_dim = 1024;     // Hidden state dimension
    const int seq_len = 1024;       // Sequence length
    const int num_sequences = 64;   // Batch size (number of sequences)
    const int output_dim = 256;     // MAX_CHAR_VALUE - vocabulary size for one-hot
    
    printf("Generating text sequence data...\n");
    printf("Input dim (embedding): %d\n", input_dim);
    printf("State dim (hidden): %d\n", state_dim);
    printf("Output dim (vocab): %d\n", output_dim);
    printf("Sequence length: %d\n", seq_len);
    printf("Number of sequences: %d\n", num_sequences);
    
    // Generate text sequence data from corpus
    float *X, *y;
    generate_text_sequence_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, 
                               "combined_corpus.txt");
    
    // Reshape data for batch processing
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped,
                                    num_sequences, seq_len, input_dim, output_dim);
    
    // Allocate device memory for input and output and copy data
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_len * num_sequences * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_len * num_sequences * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X_reshaped, seq_len * num_sequences * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y_reshaped, seq_len * num_sequences * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize state space language model
    SLM* slm = init_slm(input_dim, state_dim, output_dim, seq_len, num_sequences);
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0001f;
    
    printf("\nStarting training...\n");
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_slm(slm, d_X);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_y);

        // Print progress
        if (epoch > 0 && epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_X);
        
        // Update weights
        update_weights_slm(slm, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_slm_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_slm(slm, model_fname);
    save_text_sequence_data_to_csv(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    printf("\nTraining complete!\n");
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    cudaFree(d_X);
    cudaFree(d_y);
    free_slm(slm);
    
    return 0;
}