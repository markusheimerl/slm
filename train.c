#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <cblas.h>
#include "data.h"
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

// Generate text function using autoregressive sampling
void generate_text(SLM* slm, float temperature, const char* bos, int gen_len) {
    // Start with zero-initialized sequence
    unsigned char* tokens = (unsigned char*)calloc(slm->seq_len, sizeof(unsigned char));
    
    // Set beginning of sequence
    for (int i = 0; i < (int)strlen(bos); i++) {
        tokens[i] = (unsigned char)bos[i];
    }
    
    printf("\"%s", bos);
    fflush(stdout);
    
    // Allocate logits buffer
    float* logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = strlen(bos) - 1; pos < gen_len; pos++) {
        forward_pass_slm(slm, tokens);
        memcpy(logits, &slm->output_mlp->output[pos * slm->vocab_size], slm->vocab_size * sizeof(float));
        
        // Apply temperature and softmax
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] /= temperature;
            if (logits[v] > max_logit) max_logit = logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] = expf(logits[v] - max_logit);
            sum_exp += logits[v];
        }
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Set the next character
        tokens[pos + 1] = next_token;
        
        // Print character immediately
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

    // Parameters
    const int seq_len = 512;
    const int num_layers = 8;
    const int batch_size = 16;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    
    // Initialize or load SLM
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        slm = load_slm(argv[1], batch_size);
    } else {
        printf("Initializing new model...\n");
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    const char* corpus_file = "corpus.txt";
    size_t total_size = get_corpus_size(corpus_file);
    const size_t chunk_size = 1024 * 1024 * 1024;
    
    const float learning_rate = 0.00003f;
    int global_batch = 0;
    
    for (size_t offset = 0; offset < total_size; offset += chunk_size) {
        size_t loaded_size;
        char* chunk = load_corpus_chunk(corpus_file, offset, chunk_size, &loaded_size);
        int num_sequences = (loaded_size / seq_len);
        unsigned char* input_tokens = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
        unsigned char* target_tokens = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
        
        generate_sequences(input_tokens, target_tokens, num_sequences, seq_len, chunk, loaded_size);
        
        int num_batches = num_sequences / batch_size;
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_offset = batch * batch_size * seq_len;

            // Forward pass
            forward_pass_slm(slm, &input_tokens[batch_offset]);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, &target_tokens[batch_offset]);
            if (loss >= 7.0) raise(SIGINT);
            
            // Backward pass and update
            zero_gradients_slm(slm);
            backward_pass_slm(slm, &input_tokens[batch_offset]);
            
            // Cosine decay learning rate
            global_batch++;
            float current_lr = learning_rate * (0.5f * (1.0f + cosf(M_PI * global_batch / (float)(((total_size / chunk_size) + 1) * num_batches))));
            update_weights_slm(slm, current_lr, batch_size);
            
            printf("Batch %d, Loss: %.6f, LR: %.7f\n", global_batch, loss, current_lr);
        }
        
        free(input_tokens);
        free(target_tokens);
        free(chunk);

        printf("\n--- Sample ---\n");
        generate_text(slm, 0.9f, "Once upon a time", slm->seq_len);
        printf("--- End ---\n\n");
        
        save_slm(slm, "checkpoint_slm.bin");
    }

    // Get timestamp for filenames and save final model
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_slm(slm, model_fname);

    // Cleanup
    free_slm(slm);
    
    return 0;
}