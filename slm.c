#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "data.h"
#include "slm.h"

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    // Parse command line arguments
    char* model_file = NULL;
    if (argc > 1) {
        model_file = argv[1];
    }
    
    // Model parameters
    const int embed_dim = 512;
    const int state_dim = 512;
    const int seq_len = 1024;
    const int batch_size = 128;
    
    // Training parameters
    const int num_batches = 20000;
    const float learning_rate = 0.0001f;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("gutenberg_corpus.txt", &corpus_size);
    
    // Pre-allocate memory for sequences
    unsigned char *input_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *target_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *input_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    unsigned char *target_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    
    // Allocate GPU memory once
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, seq_len * batch_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, seq_len * batch_size * sizeof(unsigned char)));
    
    // Initialize or load model
    SLM* slm;
    if (model_file) {
        printf("Loading model from %s\n", model_file);
        slm = load_slm(model_file, batch_size);
        if (!slm) {
            printf("Failed to load model from %s\n", model_file);
            return 1;
        }
        printf("Continuing training from loaded model\n");
    } else {
        printf("Initializing new model\n");
        slm = init_slm(embed_dim, state_dim, seq_len, batch_size);
    }
    
    // Training loop
    for (int batch = 0; batch <= num_batches; batch++) {
        // Generate fresh training data from random corpus locations
        generate_char_sequences_from_corpus(&input_chars, &target_chars, 
                                          batch_size, seq_len, corpus, corpus_size);
        
        // Reshape from [batch][time] to [time][batch]
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < batch_size; b++) {
                input_reshaped[t * batch_size + b] = input_chars[b * seq_len + t];
                target_reshaped[t * batch_size + b] = target_chars[b * seq_len + t];
            }
        }
        
        // Copy to GPU
        CHECK_CUDA(cudaMemcpy(d_input_chars, input_reshaped, 
                             seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_target_chars, target_reshaped, 
                             seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_slm(slm, d_input_chars);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_target_chars);
        
        if (batch % 10 == 0) {
            printf("Batch [%d/%d], Loss: %.6f\n", batch, num_batches, loss);
        }
        
        // Generate sample text every 100 batchs
        if (batch % 100 == 0 && batch > 0) {
            printf("\n--- Sample Generation at Batch %d ---\n", batch);
            generate_text_slm(slm, "The quick brown fox", 100, 0.8f);
            generate_text_slm(slm, "Once upon a time", 100, 0.8f);
            generate_text_slm(slm, "In the beginning", 100, 0.8f);
            printf("--- End Sample Generation ---\n\n");
        }
        
        if (batch == num_batches) break;
        
        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_input_chars);
        
        // Update weights
        update_weights_slm(slm, learning_rate);
    }

    // Save model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    save_slm(slm, model_filename);
    
    // Cleanup
    free(corpus);
    free(input_chars);
    free(target_chars);
    free(input_reshaped);
    free(target_reshaped);
    cudaFree(d_input_chars);
    cudaFree(d_target_chars);
    free_slm(slm);
    
    return 0;
}