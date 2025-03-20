#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "gmlp/gpu/gmlp.h"
#include "gpu/embeddings.h"

// Helper function to sample from softmax distribution with temperature
int sample_from_distribution(float* probabilities, int vocab_size, float temperature) {
    // Apply temperature to the logits
    float* scaled_probs = (float*)malloc(vocab_size * sizeof(float));
    
    // Find the maximum value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (probabilities[i] > max_val) {
            max_val = probabilities[i];
        }
    }
    
    // Apply temperature and calculate sum for softmax
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        scaled_probs[i] = expf((probabilities[i] - max_val) / temperature);
        sum += scaled_probs[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < vocab_size; i++) {
        scaled_probs[i] /= sum;
    }
    
    // Sample from the distribution
    float random_val = (float)rand() / RAND_MAX;
    float cumulative_prob = 0.0f;
    int selected_index = vocab_size - 1;  // Default to last token if something goes wrong
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += scaled_probs[i];
        if (random_val < cumulative_prob) {
            selected_index = i;
            break;
        }
    }
    
    free(scaled_probs);
    return selected_index;
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    if (argc < 9) {
        printf("Usage: %s <embeddings.bin> <layer1.bin> <layer2.bin> <layer3.bin> <layer4.bin> <layer5.bin> <layer6.bin> \"<prompt>\" [num_tokens] [temperature]\n", argv[0]);
        return 1;
    }
    
    // Parse command-line arguments
    char* embedding_filename = argv[1];
    char* prompt = argv[8];
    
    int num_tokens = 100;  // Default: generate 100 tokens
    if (argc > 9) {
        num_tokens = atoi(argv[9]);
    }
    
    float temperature = 0.8f;  // Default temperature
    if (argc > 10) {
        temperature = atof(argv[10]);
    }
    
    // Model configuration (must match trained model)
    int context_size = 128;
    int embedding_dim = 128;
    int vocab_size = 256;
    int batch_size = 1;  // Just generate one sequence at a time
    
    // Load embeddings
    Embeddings* embeddings = load_embeddings(embedding_filename);
    if (!embeddings) {
        printf("Failed to load embeddings from %s\n", embedding_filename);
        return 1;
    }
    
    // Load gMLP layers
    GMLP* layers[6];
    for (int i = 0; i < 6; i++) {
        layers[i] = load_gmlp(argv[i+2], batch_size);
        if (!layers[i]) {
            printf("Failed to load layer %d from %s\n", i+1, argv[i+2]);
            return 1;
        }
    }
    
    // Prepare GPU memory for processing
    float* d_X_embedded_t;
    float* d_context_window;
    float* d_layer_outputs[6];
    
    // Allocate GPU memory
    checkCudaErrors(cudaMalloc(&d_X_embedded_t, batch_size * embedding_dim * sizeof(float)),
                    "Allocate d_X_embedded_t");
    checkCudaErrors(cudaMalloc(&d_context_window, batch_size * context_size * embedding_dim * sizeof(float)),
                    "Allocate d_context_window");
    
    for (int i = 0; i < 6; i++) {
        checkCudaErrors(cudaMalloc(&d_layer_outputs[i], batch_size * 
                                  ((i == 5) ? vocab_size : layers[i]->output_dim) * 
                                  sizeof(float)),
                       "Allocate d_layer_outputs");
    }
    
    // Initialize context window with zeros
    checkCudaErrors(cudaMemset(d_context_window, 0, 
                              batch_size * context_size * embedding_dim * sizeof(float)),
                    "Initialize context window");
    
    // Process prompt first
    int prompt_len = strlen(prompt);
    int context_pos = 0;
    
    printf("Prompt: %s\n", prompt);
    printf("Generating %d tokens with temperature %.2f...\n\n", num_tokens, temperature);
    
    // Buffer for the complete generated text (prompt + new tokens)
    char generated_text[10000];
    strcpy(generated_text, prompt);
    
    // Process each character in the prompt
    for (int i = 0; i < prompt_len && i < context_size; i++) {
        unsigned char token = (unsigned char)prompt[i];
        
        // Embed the token
        unsigned char* d_token;
        checkCudaErrors(cudaMalloc(&d_token, sizeof(unsigned char)),
                       "Allocate d_token");
        checkCudaErrors(cudaMemcpy(d_token, &token, sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice),
                       "Copy token to device");
        
        embeddings_forward(embeddings, d_token, d_X_embedded_t, batch_size);
        
        // Add to context window at the next position
        checkCudaErrors(cudaMemcpy(
            d_context_window + batch_size * context_pos * embedding_dim,
            d_X_embedded_t,
            batch_size * embedding_dim * sizeof(float),
            cudaMemcpyDeviceToDevice),
            "Add token to context window");
        
        context_pos = (context_pos + 1) % context_size;
        cudaFree(d_token);
    }
    
    // Main generation loop
    for (int i = 0; i < num_tokens; i++) {
        // Forward pass through all layers (flattened context window as input)
        forward_pass_gmlp(layers[0], d_context_window);
        checkCudaErrors(cudaMemcpy(d_layer_outputs[0], layers[0]->d_predictions, 
                                  batch_size * layers[0]->output_dim * sizeof(float), 
                                  cudaMemcpyDeviceToDevice),
                       "Copy layer 0 output");
        
        for (int j = 1; j < 6; j++) {
            forward_pass_gmlp(layers[j], d_layer_outputs[j-1]);
            if (j < 5) {
                checkCudaErrors(cudaMemcpy(d_layer_outputs[j], layers[j]->d_predictions, 
                                          batch_size * layers[j]->output_dim * sizeof(float), 
                                          cudaMemcpyDeviceToDevice),
                               "Copy layer output");
            }
        }
        
        // Copy the final logits (predictions) back to host
        float* logits = (float*)malloc(vocab_size * sizeof(float));
        checkCudaErrors(cudaMemcpy(logits, layers[5]->d_predictions, 
                                  vocab_size * sizeof(float), 
                                  cudaMemcpyDeviceToHost),
                       "Copy predictions to host");
        
        // Sample the next token
        int next_token = sample_from_distribution(logits, vocab_size, temperature);
        free(logits);
        
        // Add the generated token to the output
        generated_text[prompt_len + i] = (char)next_token;
        generated_text[prompt_len + i + 1] = '\0';
        
        // Embed the generated token
        unsigned char* d_token;
        checkCudaErrors(cudaMalloc(&d_token, sizeof(unsigned char)),
                       "Allocate d_token");
        checkCudaErrors(cudaMemcpy(d_token, &next_token, sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice),
                       "Copy token to device");
        
        embeddings_forward(embeddings, d_token, d_X_embedded_t, batch_size);
        
        // Shift context window (remove oldest token)
        checkCudaErrors(cudaMemcpy(
            d_context_window,
            d_context_window + batch_size * embedding_dim,
            batch_size * (context_size - 1) * embedding_dim * sizeof(float),
            cudaMemcpyDeviceToDevice),
            "Shift context window");
        
        // Add the new token embedding to the end of context window
        checkCudaErrors(cudaMemcpy(
            d_context_window + batch_size * (context_size - 1) * embedding_dim,
            d_X_embedded_t,
            batch_size * embedding_dim * sizeof(float),
            cudaMemcpyDeviceToDevice),
            "Add new token to context window");
        
        cudaFree(d_token);
        
        // Print progress
        if ((i + 1) % 10 == 0) {
            printf("Generated %d tokens\r", i + 1);
            fflush(stdout);
        }
    }
    
    // Print generated text
    printf("\n\nGenerated Text:\n%s\n", generated_text);
    
    // Clean up
    for (int i = 0; i < 6; i++) {
        free_gmlp(layers[i]);
        cudaFree(d_layer_outputs[i]);
    }
    free_embeddings(embeddings);
    cudaFree(d_X_embedded_t);
    cudaFree(d_context_window);
    
    cudaDeviceReset();
    
    return 0;
}