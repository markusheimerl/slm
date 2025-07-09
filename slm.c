#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "slm.h"

#define VOCAB_SIZE 256
#define EMBEDDING_DIM 128
#define STATE_DIM 256
#define SEQ_LEN 512      // Reduced from 1024
#define BATCH_SIZE 32    // Reduced from 64
#define MAX_BATCHES 50   // Limit number of batches to reduce memory usage

// Training step
float train_step_slm(SLM* slm, int* d_input_batch, int* d_target_batch, float learning_rate) {
    // Clear gradients
    zero_gradients_slm(slm);
    
    // Forward pass
    forward_pass_slm(slm, d_input_batch);
    
    // Calculate loss
    float loss = calculate_loss_slm(slm, d_target_batch);
    
    // Backward pass
    backward_pass_slm(slm, d_input_batch);
    
    // Update weights
    update_weights_slm(slm, learning_rate);
    
    return loss;
}

// Simple text generation
void generate_text_slm(SLM* slm, int start_token, int length) {
    int* d_input;
    float* d_embedded;
    float* d_logits;
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_embedded, slm->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, slm->vocab_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, &start_token, sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Generated text: ");
    printf("%c", (char)start_token);
    
    // Create a single-sequence SSM for generation
    SSM* gen_ssm = init_ssm(slm->embedding_dim, STATE_DIM, slm->embedding_dim, 1, 1);
    
    // Copy weights from trained model
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_A, slm->ssm->d_A, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_B, slm->ssm->d_B, STATE_DIM * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_C, slm->ssm->d_C, slm->embedding_dim * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_D, slm->ssm->d_D, slm->embedding_dim * slm->embedding_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    
    for (int i = 0; i < length - 1; i++) {
        // Embed
        embedding_forward_kernel_slm<<<1, slm->embedding_dim>>>(
            d_embedded, slm->d_embeddings, d_input, 1, 1, slm->embedding_dim
        );
        
        // SSM forward
        forward_pass_ssm(gen_ssm, d_embedded);
        
        // Project to vocab
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemv(slm->ssm->cublas_handle, CUBLAS_OP_T,
                                slm->embedding_dim, slm->vocab_size,
                                &alpha, slm->d_output_weights, slm->embedding_dim,
                                gen_ssm->d_predictions, 1,
                                &beta, d_logits, 1));
        
        // Get logits and sample
        CHECK_CUDA(cudaMemcpy(h_logits, d_logits, slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Simple argmax sampling
        int next_token = 0;
        float max_logit = h_logits[0];
        for (int j = 1; j < slm->vocab_size; j++) {
            if (h_logits[j] > max_logit) {
                max_logit = h_logits[j];
                next_token = j;
            }
        }
        
        printf("%c", (char)next_token);
        CHECK_CUDA(cudaMemcpy(d_input, &next_token, sizeof(int), cudaMemcpyHostToDevice));
    }
    printf("\n");
    
    // Cleanup
    free(h_logits);
    cudaFree(d_input);
    cudaFree(d_embedded);
    cudaFree(d_logits);
    free_ssm(gen_ssm);
}

int main() {
    srand(time(NULL));
    
    // Load data with memory limits
    int* d_all_inputs;
    int* d_all_targets;
    int num_batches;
    load_text_sequences_stream("combined_corpus.txt", &d_all_inputs, &d_all_targets, 
                              &num_batches, SEQ_LEN, BATCH_SIZE, MAX_BATCHES);
    
    // Initialize model
    SLM* slm = init_slm(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LEN, BATCH_SIZE);
    
    // Training parameters
    const int num_epochs = 50;     // Reduced from 100
    const float learning_rate = 0.001f;
    
    printf("Starting training with %d batches...\n", num_batches);
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch pointers
            int* d_input_batch = d_all_inputs + batch * BATCH_SIZE * SEQ_LEN;
            int* d_target_batch = d_all_targets + batch * BATCH_SIZE * SEQ_LEN;
            
            float loss = train_step_slm(slm, d_input_batch, d_target_batch, learning_rate);
            epoch_loss += loss;
        }
        
        epoch_loss /= num_batches;
        
        if (epoch % 5 == 0) {
            printf("Epoch %d/%d, Loss: %.4f, Perplexity: %.2f\n", 
                   epoch + 1, num_epochs, epoch_loss, expf(epoch_loss));
            
            // Generate sample
            generate_text_slm(slm, 'T', 100);
        }
    }
    
    // Save model
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    char model_path[64];
    snprintf(model_path, sizeof(model_path), "slm_%s.bin", timestamp);
    save_ssm(slm->ssm, model_path);
    
    printf("Model saved to %s\n", model_path);
    
    // Cleanup
    cudaFree(d_all_inputs);
    cudaFree(d_all_targets);
    free_slm(slm);
    
    return 0;
}