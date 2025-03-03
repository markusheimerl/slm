#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"
#include "embeddings.h"

// ---------------------------------------------------------------------
// Function: Sample from probability distribution (using categorical sampling)
// ---------------------------------------------------------------------
int sample_from_distribution(float* probs, int vocab_size, float temperature, float top_p) {
    // Apply temperature to sharpen/flatten the distribution
    if (temperature != 1.0f) {
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = powf(probs[i], 1.0f / temperature);
            sum += probs[i];
        }
        // Renormalize
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
    }
    
    // Apply nucleus (top-p) sampling if requested
    if (top_p < 1.0f) {
        // Sort probabilities (indirectly)
        int* indices = (int*)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) {
            indices[i] = i;
        }
        
        // Simple bubble sort (for small vocab this is fine)
        for (int i = 0; i < vocab_size - 1; i++) {
            for (int j = 0; j < vocab_size - i - 1; j++) {
                if (probs[indices[j]] < probs[indices[j + 1]]) {
                    int temp = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp;
                }
            }
        }
        
        // Find cutoff index for top-p
        float cumulative_prob = 0.0f;
        int cutoff_index = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumulative_prob += probs[indices[i]];
            if (cumulative_prob >= top_p) {
                cutoff_index = i + 1;
                break;
            }
        }
        
        // Set all probabilities outside top-p to zero and renormalize
        float new_sum = 0.0f;
        float* new_probs = (float*)calloc(vocab_size, sizeof(float));
        
        for (int i = 0; i < cutoff_index; i++) {
            new_probs[indices[i]] = probs[indices[i]];
            new_sum += probs[indices[i]];
        }
        
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = new_probs[i] / new_sum;
        }
        
        free(indices);
        free(new_probs);
    }
    
    // Sample from the distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r < cdf) {
            return i;
        }
    }
    
    // Fallback (should rarely happen due to floating point precision)
    return vocab_size - 1;
}

int main(int argc, char** argv) {
    // Seed random generator
    srand(time(NULL));
    
    // Files for stacked models
    char* encoder_filename = NULL;
    char* reasoning_filename = NULL;
    char* output_filename = NULL;
    char* embedding_filename = NULL;
    
    // Generation parameters
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 512;
    
    // Parse command line arguments
    if (argc >= 5) {
        encoder_filename = argv[1];
        reasoning_filename = argv[2];
        output_filename = argv[3];
        embedding_filename = argv[4];
    } else {
        fprintf(stderr, "Usage: %s <encoder_model> <reasoning_model> <output_model> <embeddings_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("=== SLM Generation with Stacked SSM ===\n");
    printf("Encoder model: %s\n", encoder_filename);
    printf("Reasoning model: %s\n", reasoning_filename);
    printf("Output model: %s\n", output_filename);
    printf("Embeddings file: %s\n", embedding_filename);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n\n", max_tokens);
    
    // Load embeddings and models
    Embeddings* embeddings = load_embeddings_inference(embedding_filename);
    SSM* encoder_ssm = load_ssm(encoder_filename);
    SSM* reasoning_ssm = load_ssm(reasoning_filename);
    SSM* output_ssm = load_ssm(output_filename);
    
    // Reset SSM states
    CHECK_CUDA(cudaMemset(encoder_ssm->d_state, 0, encoder_ssm->batch_size * encoder_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(reasoning_ssm->d_state, 0, reasoning_ssm->batch_size * reasoning_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(output_ssm->d_state, 0, output_ssm->batch_size * output_ssm->state_dim * sizeof(float)));
    
    // Allocate memory for byte inputs and intermediate data
    unsigned char* d_input_byte;
    float* d_input_embedded;
    float* d_encoder_output;
    float* d_reasoning_output;
    float* h_output_probs = (float*)malloc(embeddings->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input_byte, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_input_embedded, embeddings->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_encoder_output, encoder_ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_reasoning_output, reasoning_ssm->output_dim * sizeof(float)));
    
    // Start with the prompt
    const char* prompt = "Once upon a time";
    size_t prompt_len = strlen(prompt);
    
    // Output the prompt
    printf("%s", prompt);
    fflush(stdout);
    
    // Process each character in the prompt
    for (size_t i = 0; i < prompt_len; i++) {
        unsigned char current_byte = prompt[i];
        CHECK_CUDA(cudaMemcpy(d_input_byte, &current_byte, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Embed the byte
        embeddings_forward(embeddings, d_input_byte, d_input_embedded, 1);
        
        // Forward pass through encoder model
        forward_pass(encoder_ssm, d_input_embedded);
        
        // Copy encoder output for reasoning model
        CHECK_CUDA(cudaMemcpy(d_encoder_output, encoder_ssm->d_predictions, 
                             encoder_ssm->output_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        // Forward pass through reasoning model
        forward_pass(reasoning_ssm, d_encoder_output);
        
        // Copy reasoning output for output model
        CHECK_CUDA(cudaMemcpy(d_reasoning_output, reasoning_ssm->d_predictions, 
                             reasoning_ssm->output_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        // Forward pass through output model
        forward_pass(output_ssm, d_reasoning_output);
    }
    
    // Generate tokens
    for (int i = 0; i < max_tokens; i++) {
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 1>>>(output_ssm->d_predictions, 1, embeddings->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_output_probs, output_ssm->d_predictions, 
                             embeddings->vocab_size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Sample the next token
        int next_byte = sample_from_distribution(h_output_probs, embeddings->vocab_size, 
                                                temperature, top_p);
        
        // Print the character
        putchar(next_byte);
        fflush(stdout);
        
        // Process the next token through all models
        unsigned char current_byte = next_byte;
        CHECK_CUDA(cudaMemcpy(d_input_byte, &current_byte, sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Embed the byte
        embeddings_forward(embeddings, d_input_byte, d_input_embedded, 1);
        
        // Forward pass through all models with proper copies between stages
        forward_pass(encoder_ssm, d_input_embedded);
        CHECK_CUDA(cudaMemcpy(d_encoder_output, encoder_ssm->d_predictions, 
                            encoder_ssm->output_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        forward_pass(reasoning_ssm, d_encoder_output);
        CHECK_CUDA(cudaMemcpy(d_reasoning_output, reasoning_ssm->d_predictions, 
                            reasoning_ssm->output_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        forward_pass(output_ssm, d_reasoning_output);
    }
    
    printf("\n\n[Generation complete - reached token limit]\n");
    
    // Clean up
    free(h_output_probs);
    free_embeddings(embeddings);
    free_ssm(encoder_ssm);
    free_ssm(reasoning_ssm);
    free_ssm(output_ssm);
    cudaFree(d_input_byte);
    cudaFree(d_input_embedded);
    cudaFree(d_encoder_output);
    cudaFree(d_reasoning_output);
    
    return 0;
}