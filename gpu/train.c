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
    const int seq_len = 4096;
    const int d_model = 256;
    const int hidden_dim = 512;
    const int num_layers = 4;
    const int batch_size = 16;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("../corpus.txt", &corpus_size);
    
    // Generate random sequences from corpus
    const int num_sequences = (corpus_size - 1) / seq_len;
    unsigned char* input_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    unsigned char* target_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    generate_char_sequences_from_corpus(&input_chars, &target_chars, num_sequences, seq_len, corpus, corpus_size);
        
    // Initialize SLM
    SLM* slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    
    // Training parameters
    const int num_epochs = 10;
    const float learning_rate = 0.0003f;
    const int num_batches = num_sequences / batch_size;
    
    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
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
            
            // Print progress
            if (batch % 2 == 0) {
                printf("  Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
        }
        
        epoch_loss /= num_batches;

        // Print epoch summary
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss);
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
    
    // Show some training statistics
    printf("Total parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + seq_len * d_model + d_model * slm->vocab_size +         num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    printf("Characters processed: %d (%.1f%% of corpus)\n", num_sequences * seq_len, (float)(num_sequences * seq_len) / corpus_size * 100);
    
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