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

// Shuffle training data (Fisher-Yates)
void shuffle_data(unsigned char* input_tokens, unsigned char* target_tokens, int num_sections, int seq_len) {
    for (int i = num_sections - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap sections i and j
        for (int k = 0; k < seq_len; k++) {
            unsigned char temp = input_tokens[i * seq_len + k];
            input_tokens[i * seq_len + k] = input_tokens[j * seq_len + k];
            input_tokens[j * seq_len + k] = temp;
            
            temp = target_tokens[i * seq_len + k];
            target_tokens[i * seq_len + k] = target_tokens[j * seq_len + k];
            target_tokens[j * seq_len + k] = temp;
        }
    }
}

// Generate text function using autoregressive sampling
void generate_text(SLM* slm, float temperature, unsigned char* d_input_tokens, unsigned int seq_len) {
    // Start with zero-initialized sequence
    unsigned char* h_tokens = (unsigned char*)calloc(seq_len, sizeof(unsigned char));
    
    // Set BOS token
    const char* bos = "<|bos|>";
    for (int i = 0; i < 7; i++) {
        h_tokens[i] = (unsigned char)bos[i];
    }
    
    printf("Generating text character by character...\n");
    printf("\"<|bos|>");
    fflush(stdout);
    
    // Allocate logits buffer on host
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = 6; pos < (int)(seq_len - 1); pos++) {
        // Copy current partial sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Forward pass
        forward_pass_slm(slm, d_input_tokens);
        
        // Get logits for the current position
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[pos * slm->vocab_size], slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
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
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Set the next character
        h_tokens[pos + 1] = next_token;
        
        // Print character immediately
        printf("%c", (char)next_token);
        fflush(stdout);
    }
    
    printf("\"\n");
    
    free(h_tokens);
    free(h_logits);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int seq_len = 1024;
    const int d_model = 512;
    const int hidden_dim = 2048;
    const int num_layers = 11;
    const int batch_size = 32;
    
    // Load corpus and extract BOS-delimited sections
    size_t corpus_size;
    char* corpus = load_corpus("../corpus.txt", &corpus_size);
    
    // Create token sequences from BOS-delimited sections
    unsigned char* input_tokens = NULL;
    unsigned char* target_tokens = NULL;
    int num_sections = 0;
    
    extract_bos_sections(corpus, corpus_size, &input_tokens, &target_tokens, &num_sections, seq_len);
    
    if (num_sections == 0) {
        printf("Error: No valid sections found in corpus\n");
        free(corpus);
        return 1;
    }
    
    // Initialize or load SLM
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        slm = load_slm(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new model...\n");
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Total parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Training parameters
    const int num_epochs = 200;
    const float learning_rate = 0.00007f;
    const int num_batches = num_sections / batch_size;

    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        // Shuffle training data at the beginning of each epoch
        shuffle_data(input_tokens, target_tokens, num_sections, seq_len);
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch_offset], batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_tokens);
            if(loss >= 6.0) raise(SIGINT);
            
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            
            // Update weights
            update_weights_slm(slm, learning_rate, batch_size);
            
            // Print progress
            if (batch % 2 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
            
            if (batch > 0 && batch % 1500 == 0) {
                // Generate sample text periodically
                printf("\n--- Generating sample text (epoch %d, batch %d) ---\n", epoch, batch);
                generate_text(slm, 0.8f, d_input_tokens, seq_len);
                printf("--- End generation ---\n\n");

                // Checkpoint model periodically
                char checkpoint_fname[64];
                snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_slm.bin");
                save_slm(slm, checkpoint_fname);
            }
        }
        
        epoch_loss /= num_batches;

        // Print epoch summary
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss);
    }

    // Get timestamp for filenames and save final model
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_slm(slm, model_fname);
    
    // Cleanup
    free(corpus);
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_slm(slm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}