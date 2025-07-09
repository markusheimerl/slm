#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

int main() {
    srand(time(NULL));

    // Model hyperparameters
    const int vocab_size = 256;
    const int embedding_dim = 128;
    const int seq_len = 512;
    const int batch_size = 32;
    const int data_batches = 50;
    
    printf("Initializing SLM with vocab_size=%d, embedding_dim=%d, seq_len=%d, batch_size=%d\n",
           vocab_size, embedding_dim, seq_len, batch_size);
    
    // Load training data
    int* d_input_ids;
    int* d_target_ids;
    int num_batches;
    
    load_text_data("combined_corpus.txt", &d_input_ids, &d_target_ids, 
                   &num_batches, seq_len, batch_size, data_batches);
    
    // Initialize model
    SLM* slm = init_slm(vocab_size, embedding_dim, seq_len, batch_size);
    
    // Training parameters
    const int num_epochs = 100;
    const float learning_rate = 0.0005f;
    
    printf("Starting training for %d epochs with %d batches...\n", num_epochs, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch pointers
            int* batch_input = d_input_ids + batch * batch_size * seq_len;
            int* batch_target = d_target_ids + batch * batch_size * seq_len;
            
            // Forward pass
            forward_pass_slm(slm, batch_input);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, batch_target);
            epoch_loss += loss;
            
            // Skip weight updates on final evaluation epoch
            if (epoch == num_epochs) continue;
            
            // Backward pass and weight update
            zero_gradients_slm(slm);
            backward_pass_slm(slm, batch_input);
            update_weights_slm(slm, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.6f, Perplexity: %.2f\n", 
                   epoch, num_epochs, epoch_loss, expf(epoch_loss));
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
    
    SLM* loaded_slm = load_slm(model_filename, batch_size);
    if (loaded_slm) {
        // Test with first batch
        int* test_input = d_input_ids;
        int* test_target = d_target_ids;
        
        forward_pass_slm(loaded_slm, test_input);
        float verification_loss = calculate_loss_slm(loaded_slm, test_target);
        
        printf("Verification loss: %.6f\n", verification_loss);
        
        // Sample text generation test
        printf("\nSample predictions (first 10 tokens):\n");
        
        // Allocate host memory for predictions
        float* h_logits = (float*)malloc(seq_len * batch_size * vocab_size * sizeof(float));
        
        // Copy logits to host for inspection
        CHECK_CUDA(cudaMemcpy(h_logits, loaded_slm->d_logits, 
                             seq_len * batch_size * vocab_size * sizeof(float), 
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
            float max_logit = h_logits[t * batch_size * vocab_size];
            
            for (int v = 1; v < vocab_size; v++) {
                if (h_logits[t * batch_size * vocab_size + v] > max_logit) {
                    max_logit = h_logits[t * batch_size * vocab_size + v];
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