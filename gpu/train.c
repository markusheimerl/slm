#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "../data.h"
#include "slm.h"

SLM* slm = NULL;

// Signal handler to save model on Ctrl+C
void handle_sigint(int signum) {
    if (slm) {
        char filename[64];
        time_t now = time(NULL);
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
        save_slm(slm, filename);
    }
    exit(128 + signum);
}

// Generate text autoregressively from a prompt
void generate_text(SLM* slm, float temperature, unsigned char* d_input_tokens, const char* bos, int gen_len) {
    // Start with zero-initialized sequence
    unsigned char* h_tokens = (unsigned char*)calloc(slm->seq_len, sizeof(unsigned char));
    
    // Set beginning of sequence (prompt)
    for (int i = 0; i < (int)strlen(bos); i++) {
        h_tokens[i] = (unsigned char)bos[i];
    }
    
    printf("\"%s", bos);
    fflush(stdout);
    
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = strlen(bos) - 1; pos < gen_len; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass to get logits
        forward_pass_slm(slm, d_input_tokens);
        
        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[pos * slm->vocab_size], slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Compute softmax probabilities
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
        
        // Add sampled token to sequence
        h_tokens[pos + 1] = next_token;
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

    // Initialize cuBLAS
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Model hyperparameters
    const int seq_len = 512;
    const int num_layers = 16;
    const int batch_size = 28;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    const float learning_rate = 0.00003f;
    
    // Initialize or load model
    if (argc > 1) {
        slm = load_slm(argv[1], batch_size, cublaslt_handle);
    } else {
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Create shuffled indices for random sampling without replacement
    size_t total_sequences = (get_file_size("../corpus.txt") - 1) / seq_len;
    size_t* shuffled_indices = create_shuffled_indices(total_sequences);
    
    // Allocate host buffers for sequences
    size_t sequences_per_chunk = (128 * 1024 * 1024) / seq_len;
    unsigned char* input_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len);
    unsigned char* target_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len);
    
    // Allocate device buffers
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len));
    
    // Training loop: process corpus in chunks with random sampling
    for (size_t chunk_idx = 0; chunk_idx < total_sequences / sequences_per_chunk; chunk_idx++) {
        // Sample next chunk of sequences from shuffled corpus
        sample_sequences("../corpus.txt", &shuffled_indices[chunk_idx * sequences_per_chunk], seq_len, input_tokens, target_tokens, sequences_per_chunk);
        
        // Train on all batches in this chunk
        for (int batch = 0; batch < (int)(sequences_per_chunk / batch_size); batch++) {
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch * batch_size * seq_len], batch_size * seq_len, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch * batch_size * seq_len], batch_size * seq_len, cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_tokens);
            if (loss >= 7.0) raise(SIGINT);
            
            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            
            // Update weights with cosine learning rate schedule
            float lr = learning_rate * (0.5f * (1.0f + cosf(M_PI * ((float)((chunk_idx * (sequences_per_chunk / batch_size) + batch)) / (float)(total_sequences / batch_size)))));
            update_weights_slm(slm, lr, batch_size);
            
            printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f\n", chunk_idx, total_sequences / sequences_per_chunk, batch, (int)(sequences_per_chunk / batch_size), loss, lr);
        }
        
        // Generate sample text
        printf("\n--- Sample ---\n");
        generate_text(slm, 0.9f, d_input_tokens, "The opposite of hot is ", slm->seq_len);
        printf("--- End ---\n\n");
        
        // Save checkpoint
        save_slm(slm, "checkpoint_slm.bin");
    }
    
    // Save final model with timestamp
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_slm(slm, filename);
    
    // Cleanup
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free(shuffled_indices);
    free_slm(slm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}