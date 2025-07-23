#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "data.h"
#include "slm.h"

// Cosine annealing learning rate schedule
float cosine_schedule(float lr_init, float lr_min, int current_batch, int total_batches) {
    if (current_batch >= total_batches) return lr_min;
    
    float progress = (float)current_batch / (float)total_batches;
    float cosine_factor = 0.5f * (1.0f + cosf(M_PI * progress));
    
    return lr_min + (lr_init - lr_min) * cosine_factor;
}

// Function to calculate total model parameters
size_t calculate_model_parameters(SLM* slm) {
    size_t total_params = 0;
    total_params += slm->vocab_size * slm->embed_dim;
    SSM* ssm = slm->ssm;
    total_params += ssm->state_dim * ssm->state_dim;
    total_params += ssm->state_dim * ssm->input_dim;
    total_params += ssm->output_dim * ssm->state_dim;
    total_params += ssm->output_dim * ssm->input_dim;
    MLP* mlp = slm->mlp;
    total_params += mlp->hidden_dim * mlp->input_dim;
    total_params += mlp->output_dim * mlp->hidden_dim;
    return total_params;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    // Parse command line arguments
    char* model_file = NULL;
    if (argc > 1) {
        model_file = argv[1];
    }
    
    // Model parameters
    const int embed_dim = 256;
    const int state_dim = 128;
    const int seq_len = 4096;
    const int batch_size = 64;
    
    // Training parameters
    const int num_batches = 100000;
    const float lr_init = 0.0001f;
    const float lr_min = 0.00001f;
    
    // Pre-allocate memory for sequences
    unsigned char *input_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *target_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *input_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    unsigned char *target_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    
    // Allocate GPU memory once
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, seq_len * batch_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, seq_len * batch_size * sizeof(unsigned char)));
    
    // Initialize or load model
    SLM* slm;
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
        slm = init_slm(embed_dim, state_dim, seq_len, batch_size);
    }

    int model_size = calculate_model_parameters(slm);
    printf("Model initialized with %d parameters\n", model_size);
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("gutenberg_corpus.txt", &corpus_size, model_size * 500);

    // Training loop
    for (int batch = 0; batch <= num_batches; batch++) {
        // Calculate current learning rate using cosine schedule
        float current_lr = cosine_schedule(lr_init, lr_min, batch, num_batches);
        
        // Generate fresh training data from random corpus locations
        generate_char_sequences_from_corpus(&input_chars, &target_chars, 
                                          batch_size, seq_len, corpus, corpus_size);
        
        // Reshape from [batch][time] to [time][batch]
        for (int t = 0; t < seq_len; t++) {
            for (int b = 0; b < batch_size; b++) {
                input_reshaped[t * batch_size + b] = input_chars[b * seq_len + t];
                target_reshaped[t * batch_size + b] = target_chars[b * seq_len + t];
            }
        }
        
        // Copy to GPU
        CHECK_CUDA(cudaMemcpy(d_input_chars, input_reshaped, 
                             seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_target_chars, target_reshaped, 
                             seq_len * batch_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass_slm(slm, d_input_chars);
        
        // Calculate loss
        float loss = calculate_loss_slm(slm, d_target_chars);

        if(loss >= 5.6) {
            printf("Loss too high: %.6f, stopping training\n", loss);
            free(corpus);
            free(input_chars);
            free(target_chars);
            free(input_reshaped);
            free(target_reshaped);
            free_slm(slm);
            cudaFree(d_input_chars);
            cudaFree(d_target_chars);
            return -1;
        }
        
        if (batch % 5 == 0) {
            printf("Batch [%d/%d], Loss: %.6f, LR: %.6f\n", batch, num_batches, loss, current_lr);
        }

        // Generate sample text every 100 batches
        if (batch % 100 == 0 && batch > 0) {
            printf("\n--- Sample Generation at Batch %d ---\n", batch);
            generate_text_slm(slm, "The quick brown fox", 128, 0.8f);
            generate_text_slm(slm, "Once upon a time", 128, 0.8f);
            generate_text_slm(slm, "In the beginning", 128, 0.8f);
            printf("--- End Sample Generation ---\n\n");
        }
        
        if (batch == num_batches) break;
        
        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_input_chars);
        
        // Update weights with scheduled learning rate
        update_weights_slm(slm, current_lr);
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
    free(input_reshaped);
    free(target_reshaped);
    cudaFree(d_input_chars);
    cudaFree(d_target_chars);
    free_slm(slm);
    
    return 0;
}