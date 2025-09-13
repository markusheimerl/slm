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

// Text generation function
void generate_text(SLM* slm, char* corpus, size_t corpus_size, int length, float temperature) {
    // Start with a random seed from corpus
    int seed_start = rand() % (corpus_size - slm->seq_len - 1);
    
    // Copy seed to host buffer
    unsigned char* seed_tokens = (unsigned char*)malloc(slm->seq_len * sizeof(unsigned char));
    for (int i = 0; i < slm->seq_len; i++) seed_tokens[i] = (unsigned char)corpus[seed_start + i];
    
    printf("Seed: \"");
    for (int i = 0; i < slm->seq_len; i++) printf("%c", (char)seed_tokens[i]);
    printf("\" -> \"");
    
    // Generate text one token at a time
    for (int gen = 0; gen < length; gen++) {
        // Forward pass
        forward_pass_slm(slm, seed_tokens);
        
        // Get logits for the last position
        float* logits = &slm->output_mlp->layer_output[(slm->seq_len - 1) * slm->vocab_size];
        
        // Apply temperature and softmax
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            logits[v] /= temperature;
            if (logits[v] > max_logit) max_logit = logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            float exp_val = expf(logits[v] - max_logit);
            logits[v] = exp_val;
            sum_exp += exp_val;
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

        // Display character
        printf("%c", (char)next_token);
        fflush(stdout);
        
        // Shift sequence left and add new token
        for (int i = 0; i < slm->seq_len - 1; i++) seed_tokens[i] = seed_tokens[i + 1];
        seed_tokens[slm->seq_len - 1] = next_token;
    }
    
    printf("\"");
    free(seed_tokens);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);
    openblas_set_num_threads(4);

    // Parameters
    const int seq_len = 2048;
    const int d_model = 512;
    const int hidden_dim = 2048;
    const int num_layers = 8;
    const int batch_size = 4;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("corpus.txt", &corpus_size);
    
    // Prepare training data buffers
    const int num_sequences = (corpus_size - 1) / seq_len;
    unsigned char* input_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    unsigned char* target_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    
    // Initialize or load SLM
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        slm = load_slm(argv[1], batch_size);
    } else {
        printf("Initializing new model...\n");
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size);
    }
    
    printf("Total parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + seq_len * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Training parameters
    const int num_epochs = 100;
    const float learning_rate = 0.0003f;
    const int num_batches = num_sequences / batch_size;

    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Generate random sequences from corpus
        generate_char_sequences_from_corpus(&input_chars, &target_chars, num_sequences, seq_len, corpus, corpus_size);
        
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len;

            // Forward pass
            forward_pass_slm(slm, &input_chars[batch_offset]);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, &target_chars[batch_offset]);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, &input_chars[batch_offset]);
            
            // Update weights
            update_weights_slm(slm, learning_rate);
            
            // Print progress
            if (batch % 2 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Loss: %.6f\n", epoch, num_epochs, batch, num_batches, loss);
            }
            
            // Generate sample text periodically
            if (batch > 0 && batch % 200 == 0) {
                printf("\n--- Generated sample (epoch %d, batch %d) ---\n", epoch, batch);
                generate_text(slm, corpus, corpus_size, 128, 0.8f);
                printf("\n--- End sample ---\n\n");
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
    SLM* loaded_slm = load_slm(model_fname, batch_size);

    // Test loaded model with a forward pass
    forward_pass_slm(loaded_slm, input_chars);
    float verification_loss = calculate_loss_slm(loaded_slm, target_chars);
    printf("Verification loss: %.6f\n", verification_loss);
    
    // Cleanup
    free(corpus);
    free(input_chars);
    free(target_chars);
    free_slm(slm);
    free_slm(loaded_slm);
    
    return 0;
}