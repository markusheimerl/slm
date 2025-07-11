#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

// Text generation function
void generate_text_slm(SLM* slm, const char* seed_text, int generation_length, float temperature) {
    int seed_len = strlen(seed_text);
    if (seed_len == 0) {
        printf("Error: Empty seed text\n");
        return;
    }
    
    // Allocate temporary buffers for generation
    unsigned char* h_input = (unsigned char*)malloc(sizeof(unsigned char));
    unsigned char* d_input;
    float* h_probs = (float*)malloc(slm->vocab_size * sizeof(float));
    float* d_h_current = NULL;
    float* d_h_next = NULL;
    float* d_o_current = NULL;
    float* d_y_current = NULL;
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_h_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h_next, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_o_current, slm->ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_current, slm->vocab_size * sizeof(float)));
    
    // Initialize hidden state to zero
    CHECK_CUDA(cudaMemset(d_h_current, 0, slm->ssm->state_dim * sizeof(float)));
    
    printf("Seed: \"%s\"\nGenerated: ", seed_text);
    
    // Process seed text to build up hidden state
    for (int i = 0; i < seed_len; i++) {
        h_input[0] = (unsigned char)seed_text[i];
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = X_t B^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta, d_h_next, slm->ssm->state_dim));
        
        // H_t += H_{t-1} A^T (except for first character)
        if (i > 0) {
            CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    slm->ssm->state_dim, 1, slm->ssm->state_dim,
                                    &alpha, slm->ssm->d_A, slm->ssm->state_dim,
                                    d_h_current, slm->ssm->state_dim,
                                    &beta_add, d_h_next, slm->ssm->state_dim));
        }
        
        // Swap current and next
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;
    }
    
    // Now generate new characters
    for (int i = 0; i < generation_length; i++) {
        // Use the last character of seed (if first generation) or the last generated character
        if (i == 0) {
            h_input[0] = (unsigned char)seed_text[seed_len - 1];
        }
        
        CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Embed the character
        dim3 block(256);
        dim3 grid(1, 1);
        embedding_lookup_kernel<<<grid, block>>>(
            slm->d_embedded_input, slm->d_embeddings, d_input, 1, slm->embed_dim
        );
        
        // Forward pass for one timestep
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float beta_add = 1.0f;
        
        // H_t = X_t B^T + H_{t-1} A^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_B, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta, d_h_next, slm->ssm->state_dim));
        
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->state_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_A, slm->ssm->state_dim,
                                d_h_current, slm->ssm->state_dim,
                                &beta_add, d_h_next, slm->ssm->state_dim));
        
        // O_t = H_t σ(H_t)
        int block_size = 256;
        int num_blocks = (slm->ssm->state_dim + block_size - 1) / block_size;
        swish_forward_kernel_ssm<<<num_blocks, block_size>>>(d_o_current, d_h_next, slm->ssm->state_dim);
        
        // Y_t = O_t C^T + X_t D^T
        // Y_t = O_t C^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->state_dim,
                                &alpha, slm->ssm->d_C, slm->ssm->state_dim,
                                d_o_current, slm->ssm->state_dim,
                                &beta, d_y_current, slm->ssm->output_dim));
        
        // Y_t += X_t D^T
        CHECK_CUBLAS(cublasSgemm(slm->ssm->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                slm->ssm->output_dim, 1, slm->ssm->input_dim,
                                &alpha, slm->ssm->d_D, slm->ssm->input_dim,
                                slm->d_embedded_input, slm->ssm->input_dim,
                                &beta_add, d_y_current, slm->ssm->output_dim));
        
        // Apply softmax
        softmax_kernel<<<1, 256>>>(slm->d_softmax, d_y_current, 1, slm->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_probs, slm->d_softmax, slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling
        if (temperature != 1.0f) {
            float sum = 0.0f;
            for (int j = 0; j < slm->vocab_size; j++) {
                h_probs[j] = expf(logf(h_probs[j] + 1e-15f) / temperature);
                sum += h_probs[j];
            }
            // Renormalize
            for (int j = 0; j < slm->vocab_size; j++) {
                h_probs[j] /= sum;
            }
        }
        
        // Sample from the distribution
        float random_val = (float)rand() / (float)RAND_MAX;
        float cumulative = 0.0f;
        int sampled_char = 0;
        
        for (int j = 0; j < slm->vocab_size; j++) {
            cumulative += h_probs[j];
            if (random_val <= cumulative) {
                sampled_char = j;
                break;
            }
        }
        
        // Ensure we have a valid printable character
        if (sampled_char < 32 || sampled_char > 126) {
            sampled_char = 32; // space
        }
        
        printf("%c", sampled_char);
        fflush(stdout);
        
        // Set up for next iteration
        h_input[0] = (unsigned char)sampled_char;
        
        // Swap current and next
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;
    }
    
    printf("\n\n");
    
    // Cleanup
    free(h_input);
    free(h_probs);
    cudaFree(d_input);
    cudaFree(d_h_current);
    cudaFree(d_h_next);
    cudaFree(d_o_current);
    cudaFree(d_y_current);
}

int main() {
    srand(time(NULL));
    
    // Model parameters
    const int embed_dim = 512;
    const int state_dim = 256;
    const int vocab_size = 256;
    const int seq_len = 1024;
    const int batch_size = 128;
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0005f;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("combined_corpus.txt", &corpus_size);
    if (!corpus) {
        printf("Failed to load corpus\n");
        return 1;
    }
    
    // Pre-allocate memory for sequences
    unsigned char *input_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *target_chars = (unsigned char*)malloc(batch_size * seq_len * sizeof(unsigned char));
    unsigned char *input_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    unsigned char *target_reshaped = (unsigned char*)malloc(seq_len * batch_size * sizeof(unsigned char));
    
    // Allocate GPU memory once
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, seq_len * batch_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, seq_len * batch_size * sizeof(unsigned char)));
    
    // Initialize model
    SLM* slm = init_slm(embed_dim, state_dim, vocab_size, seq_len, batch_size);
    
    // Training loop
    for (int epoch = 0; epoch <= num_epochs; epoch++) {
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
        
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.6f\n", epoch, num_epochs, loss);
        }
        
        // Generate sample text every 100 epochs
        if (epoch % 100 == 0 && epoch > 0) {
            printf("\n--- Sample Generation at Epoch %d ---\n", epoch);
            generate_text_slm(slm, "The quick brown fox", 100, 0.8f);
            generate_text_slm(slm, "Once upon a time", 100, 0.8f);
            generate_text_slm(slm, "In the beginning", 100, 0.8f);
            printf("--- End Sample Generation ---\n\n");
        }
        
        if (epoch == num_epochs) break;
        
        // Backward pass
        zero_gradients_slm(slm);
        backward_pass_slm(slm, d_input_chars);
        
        // Update weights
        update_weights_slm(slm, learning_rate);
    }
    
    // Save model and final batch of data
    char model_file[64], data_file[64];
    time_t now = time(NULL);
    strftime(model_file, sizeof(model_file), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_file, sizeof(data_file), "%Y%m%d_%H%M%S_data.csv", localtime(&now));
    
    save_slm(slm, model_file);
    save_sequences_to_csv(input_chars, target_chars, batch_size, seq_len, data_file);
    
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