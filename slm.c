#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "slm.h"

#define VOCAB_SIZE 256
#define EMBEDDING_DIM 128
#define STATE_DIM 256
#define SEQ_LEN 512
#define BATCH_SIZE 32
#define MAX_BATCHES 50

// Reshape data for batch processing
void reshape_data_for_batch_processing(int* input_ids, int* target_ids, 
                                     int** input_reshaped, int** target_reshaped,
                                     int num_sequences, int seq_len) {
    // Reshape to: seq_len tensors of size (batch_size x 1)
    *input_reshaped = (int*)malloc(seq_len * num_sequences * sizeof(int));
    *target_reshaped = (int*)malloc(seq_len * num_sequences * sizeof(int));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time]
            int orig_idx = b * seq_len + t;
            
            // New layout: [time][seq]
            int new_idx = t * num_sequences + b;
            
            (*input_reshaped)[new_idx] = input_ids[orig_idx];
            (*target_reshaped)[new_idx] = target_ids[orig_idx];
        }
    }
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
    const int num_epochs = 100;
    const float learning_rate = 0.0005f;
    
    printf("Starting training with %d batches...\n", num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch pointers
            int* d_input_batch = d_all_inputs + batch * BATCH_SIZE * SEQ_LEN;
            int* d_target_batch = d_all_targets + batch * BATCH_SIZE * SEQ_LEN;
            
            // Forward pass
            forward_pass_slm(slm, d_input_batch);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_batch);
            epoch_loss += loss;
            
            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;
            
            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_batch);
            
            // Update weights
            update_weights_slm(slm, learning_rate);
        }
        
        epoch_loss /= num_batches;
        
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f, Perplexity: %.2f\n", 
                   epoch, num_epochs, epoch_loss, expf(epoch_loss));
        }
    }
    
    // Get timestamp for filenames
    char model_fname[64], ssm_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", 
             localtime(&now));
    strftime(ssm_fname, sizeof(ssm_fname), "%Y%m%d_%H%M%S_ssm.bin", 
             localtime(&now));
    
    // Save model and SSM with timestamped filenames
    save_slm(slm, model_fname);
    save_ssm(slm->ssm, ssm_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");
    
    // Load the model back with original batch_size
    SLM* loaded_slm = load_slm(model_fname, BATCH_SIZE);
    
    // Test with first batch
    int* d_input_batch = d_all_inputs;
    int* d_target_batch = d_all_targets;
    
    // Forward pass with loaded model
    forward_pass_slm(loaded_slm, d_input_batch);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_slm(loaded_slm, d_target_batch);
    printf("Loss with loaded model: %.8f\n", verification_loss);
    
    // Simple text generation
    printf("\nGenerating text...\n");
    
    // Copy first few tokens to host for generation
    int* h_input_sample = (int*)malloc(10 * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_input_sample, d_input_batch, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Sample input tokens: ");
    for (int i = 0; i < 10; i++) {
        printf("%c", (char)h_input_sample[i]);
    }
    printf("\n");
    
    // Allocate host memory for predictions
    float* predictions = (float*)malloc(SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float));
    
    // Copy predictions from device to host
    CHECK_CUDA(cudaMemcpy(predictions, loaded_slm->d_logits, 
                         SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Print sample predictions from first sequence
    printf("\nSample Predictions (first sequence, first 10 time steps):\n");
    printf("Time\tPredicted Token\tActual Token\n");
    printf("---------------------------------------\n");
    
    int* h_target_sample = (int*)malloc(10 * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_target_sample, d_target_batch, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int t = 0; t < 10; t++) {
        // First sequence (b=0) in reshaped format
        int idx = t * BATCH_SIZE * VOCAB_SIZE + 0 * VOCAB_SIZE;
        
        // Find argmax
        int predicted_token = 0;
        float max_logit = predictions[idx];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (predictions[idx + v] > max_logit) {
                max_logit = predictions[idx + v];
                predicted_token = v;
            }
        }
        
        int actual_token = h_target_sample[t];
        
        printf("t=%d\t%c (%d)\t\t%c (%d)\n", 
               t, (char)predicted_token, predicted_token, 
               (char)actual_token, actual_token);
    }
    
    // Calculate accuracy
    int correct = 0;
    int total = 0;
    
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            int idx = t * BATCH_SIZE * VOCAB_SIZE + b * VOCAB_SIZE;
            
            // Find argmax
            int predicted_token = 0;
            float max_logit = predictions[idx];
            for (int v = 1; v < VOCAB_SIZE; v++) {
                if (predictions[idx + v] > max_logit) {
                    max_logit = predictions[idx + v];
                    predicted_token = v;
                }
            }
            
            int actual_token_idx = t * BATCH_SIZE + b;
            int actual_token;
            CHECK_CUDA(cudaMemcpy(&actual_token, d_target_batch + actual_token_idx, 
                                 sizeof(int), cudaMemcpyDeviceToHost));
            
            if (predicted_token == actual_token) {
                correct++;
            }
            total++;
        }
    }
    
    printf("\nOverall Accuracy: %.2f%% (%d/%d)\n", 
           (float)correct * 100.0f / total, correct, total);
    
    // Cleanup
    free(h_input_sample);
    free(h_target_sample);
    free(predictions);
    cudaFree(d_all_inputs);
    cudaFree(d_all_targets);
    free_slm(slm);
    free_slm(loaded_slm);
    
    return 0;
}