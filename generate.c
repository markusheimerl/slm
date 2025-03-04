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
    char* layer1_filename = NULL;
    char* layer2_filename = NULL;
    char* layer3_filename = NULL;
    char* layer4_filename = NULL;
    char* embedding_filename = NULL;
    
    // Generation parameters
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 512;
    
    // Parse command line arguments
    if (argc >= 6) {
        layer1_filename = argv[1];
        layer2_filename = argv[2];
        layer3_filename = argv[3];
        layer4_filename = argv[4];
        embedding_filename = argv[5];
    } else {
        fprintf(stderr, "Usage: %s <layer1_model> <layer2_model> <layer3_model> <layer4_model> <embeddings_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("=== SLM Generation with Stacked SSM ===\n");
    printf("Layer 1 model: %s\n", layer1_filename);
    printf("Layer 2 model: %s\n", layer2_filename);
    printf("Layer 3 model: %s\n", layer3_filename);
    printf("Layer 4 model: %s\n", layer4_filename);
    printf("Embeddings file: %s\n", embedding_filename);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n\n", max_tokens);
    
    // Load embeddings and models
    Embeddings* embeddings = load_embeddings(embedding_filename);
    SSM* layer1_ssm = load_ssm(layer1_filename, 1);
    SSM* layer2_ssm = load_ssm(layer2_filename, 1);
    SSM* layer3_ssm = load_ssm(layer3_filename, 1);
    SSM* layer4_ssm = load_ssm(layer4_filename, 1);
    
    // Reset SSM states
    CHECK_CUDA(cudaMemset(layer1_ssm->d_state, 0, layer1_ssm->batch_size * layer1_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer2_ssm->d_state, 0, layer2_ssm->batch_size * layer2_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer3_ssm->d_state, 0, layer3_ssm->batch_size * layer3_ssm->state_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(layer4_ssm->d_state, 0, layer4_ssm->batch_size * layer4_ssm->state_dim * sizeof(float)));
    
    // Allocate memory for byte inputs and intermediate data
    unsigned char* d_input_byte;
    float* d_input_embedded;
    float* d_layer1_output;
    float* d_layer2_output;
    float* d_layer3_output;
    float* h_output_probs = (float*)malloc(embeddings->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input_byte, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_input_embedded, embeddings->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer1_output, layer1_ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer2_output, layer2_ssm->output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_layer3_output, layer3_ssm->output_dim * sizeof(float)));
    
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
        
        // Forward pass through layer 1 model
        forward_pass(layer1_ssm, d_input_embedded);
        
        // Copy layer1 output for layer2 model
        CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                             layer1_ssm->output_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 2 model
        forward_pass(layer2_ssm, d_layer1_output);
        
        // Copy layer2 output for layer3 model
        CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                             layer2_ssm->output_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 3 model
        forward_pass(layer3_ssm, d_layer2_output);
        
        // Copy layer3 output for layer4 model
        CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                             layer3_ssm->output_dim * sizeof(float), 
                             cudaMemcpyDeviceToDevice));
        
        // Forward pass through layer 4 model
        forward_pass(layer4_ssm, d_layer3_output);
    }
    
    // Generate tokens
    for (int i = 0; i < max_tokens; i++) {
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 1>>>(layer4_ssm->d_predictions, 1, embeddings->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_output_probs, layer4_ssm->d_predictions, 
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
        forward_pass(layer1_ssm, d_input_embedded);
        CHECK_CUDA(cudaMemcpy(d_layer1_output, layer1_ssm->d_predictions, 
                            layer1_ssm->output_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        forward_pass(layer2_ssm, d_layer1_output);
        CHECK_CUDA(cudaMemcpy(d_layer2_output, layer2_ssm->d_predictions, 
                            layer2_ssm->output_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        forward_pass(layer3_ssm, d_layer2_output);
        CHECK_CUDA(cudaMemcpy(d_layer3_output, layer3_ssm->d_predictions, 
                            layer3_ssm->output_dim * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        forward_pass(layer4_ssm, d_layer3_output);
    }
    
    printf("\n\n[Generation complete - reached token limit]\n");
    
    // Clean up
    free(h_output_probs);
    free_embeddings(embeddings);
    free_ssm(layer1_ssm);
    free_ssm(layer2_ssm);
    free_ssm(layer3_ssm);
    free_ssm(layer4_ssm);
    cudaFree(d_input_byte);
    cudaFree(d_input_embedded);
    cudaFree(d_layer1_output);
    cudaFree(d_layer2_output);
    cudaFree(d_layer3_output);
    
    return 0;
}