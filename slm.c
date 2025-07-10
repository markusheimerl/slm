#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

int main() {
    srand(time(NULL));
    
    // Model parameters
    const int embed_dim = 512;
    const int state_dim = 256;
    const int vocab_size = 256;
    const int seq_len = 256;
    const int batch_size = 128;
    
    // Training parameters
    const int num_epochs = 1000;
    const float learning_rate = 0.0005f;
    
    // Generate training data
    unsigned char *input_chars, *target_chars;
    generate_char_sequences(&input_chars, &target_chars, batch_size, seq_len, "combined_corpus.txt");
    
    // Move data to GPU
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, seq_len * batch_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, seq_len * batch_size * sizeof(unsigned char)));
    
    // Reshape from [batch][time] to [time][batch]
    unsigned char* input_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    unsigned char* target_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < batch_size; b++) {
            input_reshaped[t * batch_size + b] = input_chars[b * seq_len + t];
            target_reshaped[t * batch_size + b] = target_chars[b * seq_len + t];
        }
    }
    
    CHECK_CUDA(cudaMemcpy(d_input_chars, input_reshaped, 
                         seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target_chars, target_reshaped, 
                         seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Initialize model
    SLM* slm = init_slm(embed_dim, state_dim, vocab_size, seq_len, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch <= num_epochs; epoch++) {
        // Forward pass
        forward_pass_slm(slm, d_input_chars);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_target_chars);
        
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.6f\n", epoch, num_epochs, loss);
        }
        
        if (epoch == num_epochs) break;
        
        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_input_chars);
        
        // Update weights
        update_weights_slm(slm, learning_rate);
    }
    
    // Save model and data
    char model_file[64], data_file[64];
    time_t now = time(NULL);
    strftime(model_file, sizeof(model_file), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_file, sizeof(data_file), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    save_slm(slm, model_file);
    save_sequences_to_csv(input_chars, target_chars, batch_size, seq_len, data_file);
    
    // Cleanup
    free(input_chars);
    free(target_chars);
    free(input_reshaped);
    free(target_reshaped);
    cudaFree(d_input_chars);
    cudaFree(d_target_chars);
    free_slm(slm);
    
    return 0;
}