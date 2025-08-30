#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include "data.h"
#include "slm.h"

SLM* slm = NULL;

// SIGINT handler to save model and exit
void handle_sigint(int signum) {
    if (slm) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
        save_slm(slm, model_filename);
    }
    exit(128 + signum);
}

// Calculate total model parameters
size_t calculate_model_parameters(SLM* slm) {
    size_t total_params = 0;
    
    // Embeddings
    total_params += slm->vocab_size * slm->embed_dim;
    
    // Transformer layers
    for (int i = 0; i < slm->transformer->num_layers; i++) {
        // Attention weights (4 matrices of d_model x d_model each)
        total_params += 4 * slm->embed_dim * slm->embed_dim;
        
        // MLP weights
        total_params += slm->embed_dim * slm->transformer->mlp_hidden; // W1
        if (i == slm->transformer->num_layers - 1) {
            total_params += slm->transformer->mlp_hidden * slm->vocab_size; // W2 for last layer
        } else {
            total_params += slm->transformer->mlp_hidden * slm->embed_dim; // W2 for other layers
        }
    }

    return total_params;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Parse command line arguments
    char* model_file = NULL;
    if (argc > 1) model_file = argv[1];
    
    // Model parameters
    const int embed_dim = 512;
    const int seq_len = 1024;
    const int num_layers = 6;
    const int mlp_hidden = 2048;
    const int batch_size = 8;

    // Training parameters
    const int num_batches = 1000000;
    const float learning_rate = 0.0001f;
    const int acc_steps = 4;
    
    // Pre-allocate memory for sequences
    unsigned char *input_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *target_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    
    // Allocate GPU memory once
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, batch_size * seq_len * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, batch_size * seq_len * sizeof(unsigned char)));
    
    // Initialize or load model
    if (model_file) {
        printf("Loading model from %s\n", model_file);
        slm = load_slm(model_file, batch_size);
        if (!slm) {
            printf("Failed to load model from %s\n", model_file);
            return 1;
        }
        printf("Continuing training from loaded model\n");
    } else {
        printf("Initializing new model\n");
        slm = init_slm(embed_dim, seq_len, num_layers, mlp_hidden, batch_size);
    }

    size_t model_size = calculate_model_parameters(slm);
    printf("Model initialized with %zu parameters\n", model_size);
    
    // Load training corpus
    size_t corpus_size;
    char* corpus = load_corpus("corpus.txt", &corpus_size);
    if (!corpus) {
        printf("Failed to load corpus.txt\n");
        return 1;
    }

    // Training loop
    for (int batch = 0; batch <= num_batches; batch++) {
        // Generate fresh training data from random corpus locations
        generate_char_sequences_from_corpus(&input_chars, &target_chars, batch_size, seq_len, corpus, corpus_size);
        
        // Copy to GPU
        CHECK_CUDA(cudaMemcpy(d_input_chars, input_chars, batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_target_chars, target_chars, batch_size * seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_slm(slm, d_input_chars);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_target_chars);

        if(loss >= 5.6) {
            printf("Loss too high: %.6f, stopping training\n", loss);
            raise(SIGINT);
        }

        if (batch == num_batches) break;

        // Zero gradients
        if (batch % acc_steps == 0) zero_gradients_slm(slm);
        
        // Backward pass
        backward_pass_slm(slm, d_input_chars);
        
        // Update weights
        if ((batch + 1) % acc_steps == 0) update_weights_slm(slm, learning_rate);
            
        // Print progress
        if (batch % 10 == 0) {
            printf("Batch [%d/%d], Loss: %.6f, LR: %.6f\n", batch, num_batches, loss, learning_rate);
        }

        // Generate sample text
        if (batch % 500 == 0) {
            printf("\n--- Sample Generation at Batch %d ---\n", batch);
            generate_text_slm(slm, "The quick brown fox jumps over the lazy dog and then sits beside the river to watch ", 200, 0.8f, 0.9f);
            generate_text_slm(slm, "Once upon a time, in a distant kingdom, there lived a wise old king who loved to ", 200, 0.8f, 0.9f);
            generate_text_slm(slm, "Scientists at the university have discovered a new method for producing ", 200, 0.8f, 0.9f);
            printf("--- End Sample Generation ---\n\n");
        }
    }

    // Save model
    char model_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    save_slm(slm, model_filename);
    
    // Cleanup
    free(corpus);
    free(input_chars);
    free(target_chars);
    cudaFree(d_input_chars);
    cudaFree(d_target_chars);
    free_slm(slm);
    
    return 0;
}