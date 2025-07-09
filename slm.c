#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "slm.h"

// Model hyperparameters
#define VOCAB_SIZE 256
#define EMBEDDING_DIM 128
#define SEQ_LEN 512
#define BATCH_SIZE 32
#define NUM_EPOCHS 100
#define LEARNING_RATE 0.0005f
#define DATA_BATCHES 50

int main() {
    srand(time(NULL));
    
    printf("Initializing SLM with vocab_size=%d, embedding_dim=%d, seq_len=%d, batch_size=%d\n",
           VOCAB_SIZE, EMBEDDING_DIM, SEQ_LEN, BATCH_SIZE);
    
    // Load training data
    int* d_input_ids;
    int* d_target_ids;
    int num_batches;
    
    load_text_data("combined_corpus.txt", &d_input_ids, &d_target_ids, 
                   &num_batches, SEQ_LEN, BATCH_SIZE, DATA_BATCHES);
    
    // Initialize model
    SLM* slm = init_slm(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LEN, BATCH_SIZE);
    
    printf("Starting training for %d epochs with %d batches...\n", NUM_EPOCHS, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch <= NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch pointers
            int* batch_input = d_input_ids + batch * BATCH_SIZE * SEQ_LEN;
            int* batch_target = d_target_ids + batch * BATCH_SIZE * SEQ_LEN;
            
            // Forward pass
            forward_pass_slm(slm, batch_input);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, batch_target);
            epoch_loss += loss;
            
            // Skip weight updates on final evaluation epoch
            if (epoch == NUM_EPOCHS) continue;
            
            // Backward pass and weight update
            zero_gradients_slm(slm);
            backward_pass_slm(slm, batch_input);
            update_weights_slm(slm, LEARNING_RATE);
        }
        
        epoch_loss /= num_batches;
        
        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.6f, Perplexity: %.2f\n", 
                   epoch, NUM_EPOCHS, epoch_loss, expf(epoch_loss));
        }
    }
    
    // Generate timestamped filenames
    char model_filename[64], ssm_filename[64];
    time_t now = time(NULL);
    strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    strftime(ssm_filename, sizeof(ssm_filename), "%Y%m%d_%H%M%S_ssm.bin", localtime(&now));
    
    // Save models
    save_slm(slm, model_filename);
    save_ssm(slm->ssm, ssm_filename);
    
    // Test model loading and verification
    printf("\nTesting model loading and verification...\n");
    
    SLM* loaded_slm = load_slm(model_filename, BATCH_SIZE);
    if (loaded_slm) {
        // Test with first batch
        int* test_input = d_input_ids;
        int* test_target = d_target_ids;
        
        forward_pass_slm(loaded_slm, test_input);
        float verification_loss = calculate_loss_slm(loaded_slm, test_target);
        
        printf("Verification loss: %.6f\n", verification_loss);
        
        // Sample text generation test
        printf("\nSample predictions (first 10 tokens):\n");
        
        // Copy logits to host for inspection
        float* h_logits = (float*)malloc(SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_logits, loaded_slm->d_logits, 
                             SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Copy input and target tokens to host
        int* h_input = (int*)malloc(10 * sizeof(int));
        int* h_target = (int*)malloc(10 * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_input, test_input, 10 * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_target, test_target, 10 * sizeof(int), cudaMemcpyDeviceToHost));
        
        printf("Pos\tInput\tTarget\tPredicted\n");
        printf("---\t-----\t------\t---------\n");
        
        for (int t = 0; t < 10; t++) {
            // Find predicted token (argmax)
            int pred_token = 0;
            float max_logit = h_logits[t * BATCH_SIZE * VOCAB_SIZE];
            
            for (int v = 1; v < VOCAB_SIZE; v++) {
                if (h_logits[t * BATCH_SIZE * VOCAB_SIZE + v] > max_logit) {
                    max_logit = h_logits[t * BATCH_SIZE * VOCAB_SIZE + v];
                    pred_token = v;
                }
            }
            
            printf("%d\t'%c'\t'%c'\t'%c'\n", t, 
                   (char)h_input[t], (char)h_target[t], (char)pred_token);
        }
        
        free(h_logits);
        free(h_input);
        free(h_target);
        free_slm(loaded_slm);
    }
    
    // Cleanup
    cudaFree(d_input_ids);
    cudaFree(d_target_ids);
    free_slm(slm);
    
    printf("\nTraining completed successfully!\n");
    
    return 0;
}