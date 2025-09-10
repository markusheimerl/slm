#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "slm.h"

int main() {
    srand(time(NULL));

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 128;
    const int d_model = 256;
    const int hidden_dim = 1024;
    const int num_layers = 6;
    const int batch_size = 32;
    const bool is_causal = true;  // Causal attention for language modeling
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("../corpus.txt", &corpus_size);
    if (!corpus) {
        CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
        return 1;
    }
    
    // Calculate number of sequences we can extract
    const int num_sequences = 8192;  // Total sequences for training
    
    // Allocate memory for character sequences
    unsigned char* input_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    unsigned char* target_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    
    if (!input_chars || !target_chars) {
        printf("Error: Could not allocate memory for sequences\n");
        free(corpus);
        CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
        return 1;
    }
    
    // Generate character sequences from corpus
    generate_char_sequences_from_corpus(&input_chars, &target_chars, num_sequences, seq_len, corpus, corpus_size);
    
    // Initialize SLM
    SLM* slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, cublaslt_handle);
    
    // Training parameters
    const int num_epochs = 10;
    const float learning_rate = 0.0003f;
    const int num_batches = num_sequences / batch_size;
    
    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
    printf("Starting training...\n");
    printf("Corpus size: %zu characters\n", corpus_size);
    printf("Training sequences: %d\n", num_sequences);
    printf("Batch size: %d, Batches per epoch: %d\n", batch_size, num_batches);
    printf("Model parameters: seq_len=%d, d_model=%d, hidden_dim=%d, num_layers=%d\n", 
           seq_len, d_model, hidden_dim, num_layers);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_chars[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_chars[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_tokens);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            
            // Update weights
            update_weights_slm(slm, learning_rate);
            
            // Print progress during training
            if (batch % (num_batches / 4) == 0) {
                printf("  Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", 
                       epoch, num_epochs, batch, num_batches, loss);
            }
        }
        
        epoch_loss /= num_batches;

        // Print epoch summary
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss);
        
        // Generate sample text every few epochs
        if (epoch % 3 == 0 && epoch > 0) {
            printf("\n--- Sample Generation ---\n");
            unsigned char seed[] = "The quick brown";
            //generate_text_slm(slm, seed, strlen((char*)seed), 100, 0.8f);
        }
    }

    // Get timestamp for filenames
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));

    // Save model with timestamped filename
    save_slm(slm, model_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    SLM* loaded_slm = load_slm(model_fname, batch_size, cublaslt_handle);

    // Test loaded model with a forward pass
    CHECK_CUDA(cudaMemcpy(d_input_tokens, input_chars, batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target_tokens, target_chars, batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    forward_pass_slm(loaded_slm, d_input_tokens);
    float verification_loss = calculate_loss_slm(loaded_slm, d_target_tokens);
    printf("Verification loss: %.6f\n", verification_loss);
    
    // Generate sample text with loaded model
    printf("\n--- Final Text Generation ---\n");
    unsigned char final_seed[] = "Once upon a time";
    // generate_text_slm(loaded_slm, final_seed, strlen((char*)final_seed), 200, 0.7f);
    
    // Show some training statistics
    printf("\n--- Training Statistics ---\n");
    printf("Total parameters: ~%.1fM\n", 
           (float)(VOCAB_SIZE * d_model + seq_len * d_model + d_model * VOCAB_SIZE + 
                   num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    printf("Training completed successfully!\n");
    
    // Cleanup
    free(corpus);
    free(input_chars);
    free(target_chars);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_slm(slm);
    free_slm(loaded_slm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}