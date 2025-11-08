#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <cblas.h>
#include "data.h"
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
void generate_text(SLM* slm, float temperature, const char* bos, int gen_len) {
    // Start with zero-initialized sequence
    unsigned char* tokens = (unsigned char*)calloc(slm->seq_len, sizeof(unsigned char));
    
    // Set beginning of sequence (prompt)
    for (int i = 0; i < (int)strlen(bos); i++) {
        tokens[i] = (unsigned char)bos[i];
    }
    
    printf("\"%s", bos);
    fflush(stdout);
    
    float* logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = strlen(bos) - 1; pos < gen_len; pos++) {
        // Forward pass to get logits
        forward_pass_slm(slm, tokens);
        memcpy(logits, &slm->output_mlp->output[pos * slm->vocab_size], slm->vocab_size * sizeof(float));
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] /= temperature;
            if (logits[v] > max_logit) max_logit = logits[v];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] = expf(logits[v] - max_logit);
            sum_exp += logits[v];
        }
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Add sampled token to sequence
        tokens[pos + 1] = next_token;
        printf("%c", (char)next_token);
        fflush(stdout);
    }
    
    printf("\"\n");
    free(tokens);
    free(logits);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);
    openblas_set_num_threads(4);

    // Model hyperparameters
    const int seq_len = 512;
    const int num_layers = 8;
    const int batch_size = 16;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    const float learning_rate = 0.00003f;
    
    // Initialize or load model
    if (argc > 1) {
        slm = load_slm(argv[1], batch_size);
    } else {
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Create shuffled indices for random sampling without replacement
    size_t total_sequences = (get_file_size("corpus.txt") - 1) / seq_len;
    size_t* shuffled_indices = create_shuffled_indices(total_sequences);
    
    // Allocate buffers for sequences
    size_t sequences_per_chunk = (128 * 1024 * 1024) / seq_len;
    unsigned char* input_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len);
    unsigned char* target_tokens = (unsigned char*)malloc(sequences_per_chunk * seq_len);
    
    // Training loop: process corpus in chunks with random sampling
    for (size_t chunk_idx = 0; chunk_idx < total_sequences / sequences_per_chunk; chunk_idx++) {
        // Sample next chunk of sequences from shuffled corpus
        sample_sequences("corpus.txt", &shuffled_indices[chunk_idx * sequences_per_chunk], seq_len, input_tokens, target_tokens, sequences_per_chunk);
        
        // Train on all batches in this chunk
        for (int batch = 0; batch < (int)(sequences_per_chunk / batch_size); batch++) {
            // Forward pass
            forward_pass_slm(slm, &input_tokens[batch * batch_size * seq_len]);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, &target_tokens[batch * batch_size * seq_len]);
            if (loss >= 7.0) raise(SIGINT);
            
            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, &input_tokens[batch * batch_size * seq_len]);
            
            // Update weights with cosine learning rate schedule
            float lr = learning_rate * (0.5f * (1.0f + cosf(M_PI * ((float)((chunk_idx * (sequences_per_chunk / batch_size) + batch)) / (float)(total_sequences / batch_size)))));
            update_weights_slm(slm, lr, batch_size);
            
            printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f\n", chunk_idx, total_sequences / sequences_per_chunk, batch, (int)(sequences_per_chunk / batch_size), loss, lr);
        }
        
        // Generate sample text
        printf("\n--- Sample ---\n");
        generate_text(slm, 0.9f, "The opposite of hot is ", slm->seq_len);
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
    free(shuffled_indices);
    free_slm(slm);
    
    return 0;
}