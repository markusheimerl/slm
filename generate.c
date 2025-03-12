#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"
#include "gpu/embeddings.h"

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
    
    // Generation parameters
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 512;
    
    // Parse command line arguments
    if (argc < 26) {
        fprintf(stderr, "Usage: %s <layer1_model> <layer2_model> ... <layer24_model> <embeddings_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("=== SLM Generation with 24-Layer SSM ===\n");
    for (int i = 0; i < 24; i++) {
        printf("Layer %2d model: %s\n", i+1, argv[i+1]);
    }
    printf("Embeddings file: %s\n", argv[25]);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n\n", max_tokens);
    
    // Load embeddings and models
    Embeddings* embeddings = load_embeddings(argv[25]);
    SSM* layers[24];
    
    for (int i = 0; i < 24; i++) {
        layers[i] = load_ssm(argv[i+1], 1);
    }
    
    // Reset SSM states
    for (int i = 0; i < 24; i++) {
        CHECK_CUDA(cudaMemset(layers[i]->d_state, 0, 
                             layers[i]->batch_size * layers[i]->state_dim * sizeof(float)));
    }
    
    // Allocate memory for byte inputs and intermediate data
    unsigned char* d_input_byte;
    float* d_input_embedded;
    float* d_layer_outputs[24];
    float* h_output_probs = (float*)malloc(embeddings->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input_byte, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_input_embedded, embeddings->embedding_dim * sizeof(float)));
    
    for (int i = 0; i < 24; i++) {
        CHECK_CUDA(cudaMalloc(&d_layer_outputs[i], layers[i]->output_dim * sizeof(float)));
    }
    
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
        
        // Forward pass through all layers
        forward_pass_ssm(layers[0], d_input_embedded);
        CHECK_CUDA(cudaMemcpy(d_layer_outputs[0], layers[0]->d_predictions, 
                           layers[0]->output_dim * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        for (int j = 1; j < 24; j++) {
            forward_pass_ssm(layers[j], d_layer_outputs[j-1]);
            if (j < 23) {  // Don't need to copy for the last layer
                CHECK_CUDA(cudaMemcpy(d_layer_outputs[j], layers[j]->d_predictions, 
                                   layers[j]->output_dim * sizeof(float), 
                                   cudaMemcpyDeviceToDevice));
            }
        }
    }
    
    // Generate tokens
    for (int i = 0; i < max_tokens; i++) {
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 1>>>(layers[23]->d_predictions, 1, embeddings->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_output_probs, layers[23]->d_predictions, 
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
        
        // Forward pass through all layers
        embeddings_forward(embeddings, d_input_byte, d_input_embedded, 1);
        
        forward_pass_ssm(layers[0], d_input_embedded);
        CHECK_CUDA(cudaMemcpy(d_layer_outputs[0], layers[0]->d_predictions, 
                           layers[0]->output_dim * sizeof(float), 
                           cudaMemcpyDeviceToDevice));
        
        for (int j = 1; j < 24; j++) {
            forward_pass_ssm(layers[j], d_layer_outputs[j-1]);
            if (j < 23) {  // Don't need to copy for the last layer
                CHECK_CUDA(cudaMemcpy(d_layer_outputs[j], layers[j]->d_predictions, 
                                   layers[j]->output_dim * sizeof(float), 
                                   cudaMemcpyDeviceToDevice));
            }
        }
    }
    
    printf("\n\n[Generation complete - reached token limit]\n");
    
    // Clean up
    free(h_output_probs);
    free_embeddings(embeddings);
    
    for (int i = 0; i < 24; i++) {
        free_ssm(layers[i]);
        cudaFree(d_layer_outputs[i]);
    }
    
    cudaFree(d_input_byte);
    cudaFree(d_input_embedded);
    
    return 0;
}