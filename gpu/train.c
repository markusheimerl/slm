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
void generate_text(SLM* slm, float temperature, unsigned char* d_input_tokens, unsigned int seq_len, const char* start_token) {
    // Start with zero-initialized sequence
    unsigned char* h_tokens = (unsigned char*)calloc(seq_len, sizeof(unsigned char));
    
    int start_len = strlen(start_token);
    
    // Set starting token
    for (int i = 0; i < start_len; i++) h_tokens[i] = (unsigned char)start_token[i];
    
    printf("Generating text character by character...\n");
    printf("\"%s", start_token);
    fflush(stdout);
    
    // Allocate logits buffer on host
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = start_len - 1; pos < (int)(seq_len - 1); pos++) {
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
    const int seq_len = 512;
    const int num_layers = 32;
    const int batch_size = 24;
    const int d_model = num_layers * 16;
    const int hidden_dim = d_model * 4;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("../corpus.txt", &corpus_size);
    
    // Create token sequences
    unsigned char* input_tokens = NULL;
    unsigned char* target_tokens = NULL;
    int num_sections = 0;
    
    extract_sections(corpus, corpus_size, &input_tokens, &target_tokens, &num_sections, seq_len);
    
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
    
    long long total_params = slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4LL * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model);
    printf("Total parameters: ~%.1fM\n", (float)total_params / 1e6f);
    
    // Calculate optimal training tokens using Chinchilla scaling laws
    long long dataset_tokens = (long long)num_sections * seq_len;
    long long optimal_tokens = 20LL * total_params;
    int num_epochs = (int)((optimal_tokens + dataset_tokens - 1) / dataset_tokens);
    if (num_epochs < 1) num_epochs = 1;
    
    // Training parameters
    const float max_learning_rate = 0.00004f;
    const float min_learning_rate = 0.000004f;
    const int num_batches = num_sections / batch_size;
    const int total_batches = num_epochs * num_batches;

    // Allocate device memory for batch data
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned char)));
    
    // Timing for ETA
    time_t training_start = time(NULL);
    
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
            if(loss >= 10.0) raise(SIGINT);
            
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Cosine learning rate decay
            int current_batch = epoch * num_batches + batch;
            float progress = (float)current_batch / (float)total_batches;
            float learning_rate = min_learning_rate + 0.5f * (max_learning_rate - min_learning_rate) * (1.0f + cosf(progress * M_PI));

            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            
            // Update weights
            update_weights_slm(slm, learning_rate, batch_size);
            
            // Print progress
            printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            
            // Print ETA
            if ((batch + 1) % 10 == 0) {
                time_t current_time = time(NULL);
                double elapsed = difftime(current_time, training_start);
                int total_batches_done = epoch * num_batches + batch + 1;
                double avg_time_per_batch = elapsed / total_batches_done;
                double eta_seconds = avg_time_per_batch * (total_batches - total_batches_done);
                int eta_hours = (int)(eta_seconds / 3600);
                int eta_minutes = (int)((eta_seconds - eta_hours * 3600) / 60);
                printf(">>> ETA: %dh %dm remaining <<<\n", eta_hours, eta_minutes);
                printf(">>> LR: %.7f <<<\n", learning_rate);
            }
        }

        // Generate sample text
        const char* token_types[] = {"<|wiki|>", "<|web|>", "<|story|>", "<|talk|>"};
        const char* type_names[] = {"wiki", "web", "story", "conversation"};
        
        for (int i = 0; i < 4; i++) {
            printf("\n--- Generating sample %s text ---\n", type_names[i]);
            generate_text(slm, 0.8f, d_input_tokens, seq_len, token_types[i]);
            printf("--- End generation ---\n\n");
        }

        // Checkpoint model
        char checkpoint_fname[64];
        snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_slm.bin");
        save_slm(slm, checkpoint_fname);
        
        // Print epoch summary
        printf("Epoch [%d/%d] completed, Average Loss: %.6f\n", epoch, num_epochs, epoch_loss / num_batches);
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