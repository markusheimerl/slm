#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "../data.h"
#include "slm.h"

SLM* slm = NULL;

// SIGINT handler to save model and exit
void handle_sigint(int signum) {
    if (slm) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
        save_slm(slm, model_filename);
    }
    exit(128 + signum);
}

// Text generation function
void generate_text(SLM* slm, uint32_t* encoded_corpus, uint32_t corpus_tokens, int length, float temperature, uint32_t* d_input_tokens) {
    // Start with a random seed from corpus
    int seed_start = rand() % (corpus_tokens - slm->seq_len - 1);
    
    // Copy seed to host buffer
    uint32_t* h_seed_tokens = (uint32_t*)malloc(slm->seq_len * sizeof(uint32_t));
    for (int i = 0; i < slm->seq_len; i++) h_seed_tokens[i] = encoded_corpus[seed_start + i];
    
    // Decode seed
    char* seed_text = decode_bpe(slm->bpe, h_seed_tokens, slm->seq_len);
    printf("Seed: \"%s\" -> \"", seed_text);
    free(seed_text);
    
    // Copy seed to d_input_tokens
    CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Allocate logits buffer
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate text one token at a time
    for (int gen = 0; gen < length; gen++) {
        // Forward pass
        forward_pass_slm(slm, d_input_tokens);
        
        // Get logits
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[(slm->seq_len - 1) * slm->vocab_size], slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature and softmax
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            float exp_val = expf(h_logits[v] - max_logit);
            h_logits[v] = exp_val;
            sum_exp += exp_val;
        }
        
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        uint32_t next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }

        // Decode and display token
        char* decoded = decode_bpe(slm->bpe, &next_token, 1);
        printf("%s", decoded);
        fflush(stdout);
        free(decoded);
        
        // Shift sequence left and add new token
        for (int i = 0; i < slm->seq_len - 1; i++) h_seed_tokens[i] = h_seed_tokens[i + 1];
        h_seed_tokens[slm->seq_len - 1] = next_token;
        
        // Update with the new sequence
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    
    printf("\"");
    free(h_seed_tokens);
    free(h_logits);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 2048;
    const int d_model = 1024;
    const int hidden_dim = 4096;
    const int num_layers = 32;
    const int batch_size = 2;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("../corpus.txt", &corpus_size);
    
    // Initialize or load SLM (this also loads BPE)
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        slm = load_slm(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new model...\n");
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    if (!slm) {
        printf("Failed to initialize model\n");
        free(corpus);
        return 1;
    }
    
    // Encode corpus once with BPE
    printf("Encoding corpus with BPE...\n");
    uint32_t corpus_tokens;
    uint32_t* encoded_corpus = encode_bpe(slm->bpe, corpus, corpus_size, &corpus_tokens);
    printf("Corpus: %zu bytes -> %u tokens (%.1f%% compression)\n", 
           corpus_size, corpus_tokens, 100.0 * (1.0 - (double)corpus_tokens / corpus_size));
    
    // Prepare training data buffers
    const int num_sequences = (corpus_tokens - 1) / seq_len;
    uint32_t* input_tokens = (uint32_t*)malloc(num_sequences * seq_len * sizeof(uint32_t));
    uint32_t* target_tokens = (uint32_t*)malloc(num_sequences * seq_len * sizeof(uint32_t));
    
    printf("Total parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + seq_len * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Training parameters
    const int num_epochs = 100;
    const float learning_rate = 0.00001f;
    const int num_batches = num_sequences / batch_size;

    // Allocate device memory for batch data
    uint32_t *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(uint32_t)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Generate random sequences from encoded corpus
        int max_sequences = (corpus_tokens - 1) / seq_len;
        int* sequence_positions = (int*)malloc(max_sequences * sizeof(int));
        for (int i = 0; i < max_sequences; i++) {
            sequence_positions[i] = i * seq_len;
        }
        
        // Fisher-Yates shuffle
        for (int i = max_sequences - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = sequence_positions[i];
            sequence_positions[i] = sequence_positions[j];
            sequence_positions[j] = temp;
        }
        
        // Fill sequences
        for (int seq = 0; seq < num_sequences; seq++) {
            uint32_t start_pos = sequence_positions[seq];
            for (int t = 0; t < seq_len; t++) {
                int idx = seq * seq_len + t;
                input_tokens[idx]  = encoded_corpus[start_pos + t];
                target_tokens[idx] = encoded_corpus[start_pos + t + 1];
            }
        }
        free(sequence_positions);
        
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch_offset], batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch_offset], batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_tokens);
            if(loss >= 8.0) raise(SIGINT);
            
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
                printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
            
            // Generate sample text periodically
            if (batch > 0 && batch % 200 == 0) {
                printf("\n--- Generated sample (epoch %d, batch %d) ---\n", epoch, batch);
                generate_text(slm, encoded_corpus, corpus_tokens, 128, 0.8f, d_input_tokens);
                printf("\n--- End sample ---\n\n");
            }

            // Checkpoint model periodically
            if (batch > 0 && batch % 1000 == 0) {
                char checkpoint_fname[64];
                snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_slm.bin");
                save_slm(slm, checkpoint_fname);
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
    CHECK_CUDA(cudaMemcpy(d_input_tokens, input_tokens, batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target_tokens, target_tokens, batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    forward_pass_slm(loaded_slm, d_input_tokens);
    float verification_loss = calculate_loss_slm(loaded_slm, d_target_tokens);
    printf("Verification loss: %.6f\n", verification_loss);
    
    // Cleanup
    free(corpus);
    free(encoded_corpus);
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_slm(slm);
    free_slm(loaded_slm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}