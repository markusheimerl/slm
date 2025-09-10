#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "slm.h"

// Simple text generation function
void generate_text(SLM* slm, char* corpus, size_t corpus_size, int gen_length, float temperature_scale, unsigned char* d_input_tokens) {
    // Start with a random seed from corpus
    int seed_start = rand() % (corpus_size - slm->seq_len - 1);
    
    // Copy seed to host buffer
    unsigned char* h_seed_tokens = (unsigned char*)malloc(slm->seq_len * sizeof(unsigned char));
    for (int i = 0; i < slm->seq_len; i++) {
        h_seed_tokens[i] = (unsigned char)corpus[seed_start + i];
    }
    
    // Copy seed to batch 0 of d_input_tokens
    CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    printf("Seed: \"");
    for (int i = 0; i < slm->seq_len; i++) {
        printf("%c", (char)h_seed_tokens[i]);
    }
    printf("\" -> \"");
    
    // Generate text one token at a time
    for (int gen = 0; gen < gen_length; gen++) {
        // Forward pass
        forward_pass_slm(slm, d_input_tokens);
        
        // Get logits for batch 0, last position
        float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
        int last_pos_offset = (slm->seq_len - 1) * slm->vocab_size; // batch 0, last position
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->d_logits[last_pos_offset], 
                             slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature and softmax
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature_scale;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }

        // Print only valid characters
        if(next_token < 32 || next_token > 126) next_token = (unsigned char)' ';
        
        printf("%c", (char)next_token);
        fflush(stdout);
        
        // Shift sequence left and add new token
        for (int i = 0; i < slm->seq_len - 1; i++) {
            h_seed_tokens[i] = h_seed_tokens[i + 1];
        }
        h_seed_tokens[slm->seq_len - 1] = next_token;
        
        // Update only batch 0 with the new sequence
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        free(h_logits);
    }
    
    printf("\"\n\n");
    free(h_seed_tokens);
}

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
            
            // Generate sample text periodically
            if (batch > 0 && batch % 200 == 0) {
                printf("\n--- Generated sample (epoch %d, batch %d) ---\n", epoch, batch);
                generate_text(slm, corpus, corpus_size, 128, 0.8f, d_input_tokens);
                printf("--- End sample ---\n\n");
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
    printf("Total parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + seq_len * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
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