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
    
    // Default model file paths (will be overridden by command line args)
    char* model_filename = NULL;
    char* embedding_filename = NULL;
    
    // Generation parameters (fixed, sensible defaults)
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 512;
    
    // Parse command line arguments (model and embedding files only)
    if (argc >= 3) {
        model_filename = argv[1];
        embedding_filename = argv[2];
    } else {
        // Find the most recent model and embeddings files if not specified
        FILE* fp = popen("ls -t *_slm.bin | head -1", "r");
        if (fp) {
            char buffer[256];
            if (fgets(buffer, sizeof(buffer), fp) != NULL) {
                size_t len = strlen(buffer);
                if (len > 0 && buffer[len-1] == '\n')
                    buffer[len-1] = '\0';
                model_filename = strdup(buffer);
            }
            pclose(fp);
        }
        
        fp = popen("ls -t *_embeddings.bin | head -1", "r");
        if (fp) {
            char buffer[256];
            if (fgets(buffer, sizeof(buffer), fp) != NULL) {
                size_t len = strlen(buffer);
                if (len > 0 && buffer[len-1] == '\n')
                    buffer[len-1] = '\0';
                embedding_filename = strdup(buffer);
            }
            pclose(fp);
        }
    }
    
    if (!model_filename || !embedding_filename) {
        fprintf(stderr, "Error: Missing model or embeddings file\n");
        fprintf(stderr, "Usage: %s <model_file> <embeddings_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("=== SLM Generation ===\n");
    printf("Model file: %s\n", model_filename);
    printf("Embeddings file: %s\n", embedding_filename);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n\n", max_tokens);
    
    // Load embeddings and model
    Embeddings* embeddings = load_embeddings_inference(embedding_filename);
    if (!embeddings) {
        fprintf(stderr, "Failed to load embeddings\n");
        return EXIT_FAILURE;
    }
    
    SSM* ssm = load_ssm(model_filename);
    if (!ssm) {
        fprintf(stderr, "Failed to load model\n");
        free_embeddings(embeddings);
        return EXIT_FAILURE;
    }
    
    // Check model configuration
    if (ssm->batch_size != 1) {
        fprintf(stderr, "Model was not saved with batch_size=1 for inference\n");
        free_embeddings(embeddings);
        free_ssm(ssm);
        return EXIT_FAILURE;
    }
    
    if (ssm->input_dim != embeddings->embedding_dim) {
        fprintf(stderr, "Embedding dimension (%d) doesn't match model input dimension (%d)\n",
                embeddings->embedding_dim, ssm->input_dim);
        free_embeddings(embeddings);
        free_ssm(ssm);
        return EXIT_FAILURE;
    }
    
    if (ssm->output_dim != embeddings->vocab_size) {
        fprintf(stderr, "Model output dimension (%d) doesn't match vocabulary size (%d)\n",
                ssm->output_dim, embeddings->vocab_size);
        free_embeddings(embeddings);
        free_ssm(ssm);
        return EXIT_FAILURE;
    }
    
    // Reset SSM state
    CHECK_CUDA(cudaMemset(ssm->d_state, 0, ssm->batch_size * ssm->state_dim * sizeof(float)));
    
    // Allocate memory for input byte and embedded input
    unsigned char* d_input_byte;
    float* d_input_embedded;
    float* h_output_probs = (float*)malloc(embeddings->vocab_size * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input_byte, sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_input_embedded, embeddings->embedding_dim * sizeof(float)));
    
    // Start with the prompt
    const char* prompt = "<USER> Share a story about a garden where emotions grow as flowers. <ASSISTANT> ";
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
        
        // Forward pass through the model
        forward_pass(ssm, d_input_embedded);
    }
    
    // Generate tokens
    for (int i = 0; i < max_tokens; i++) {
        // Apply softmax to get probabilities
        softmax_kernel<<<1, 1>>>(ssm->d_predictions, 1, embeddings->vocab_size);
        
        // Copy probabilities to host
        CHECK_CUDA(cudaMemcpy(h_output_probs, ssm->d_predictions, 
                             embeddings->vocab_size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Sample the next token
        int next_byte = sample_from_distribution(h_output_probs, embeddings->vocab_size, 
                                                temperature, top_p);
        
        // Print the character
        putchar(next_byte);
        fflush(stdout);
        
        // Check for special sequences to end generation
        if (i > 0 && next_byte == '<') {
            char potential_end_tag[12] = {0};
            potential_end_tag[0] = '<';
            
            // Keep going for enough characters to potentially have "<USER>"
            for (int j = 1; j < 6 && i + j < max_tokens; j++) {
                // Apply softmax
                softmax_kernel<<<1, 1>>>(ssm->d_predictions, 1, embeddings->vocab_size);
                
                // Copy probabilities to host
                CHECK_CUDA(cudaMemcpy(h_output_probs, ssm->d_predictions, 
                                     embeddings->vocab_size * sizeof(float), 
                                     cudaMemcpyDeviceToHost));
                
                // Sample the next token
                int tag_byte = sample_from_distribution(h_output_probs, embeddings->vocab_size, 
                                                      temperature, top_p);
                
                // Update the tag
                potential_end_tag[j] = tag_byte;
                putchar(tag_byte);
                fflush(stdout);
                
                // Update for next token
                unsigned char current_byte = tag_byte;
                CHECK_CUDA(cudaMemcpy(d_input_byte, &current_byte, sizeof(unsigned char), cudaMemcpyHostToDevice));
                embeddings_forward(embeddings, d_input_byte, d_input_embedded, 1);
                forward_pass(ssm, d_input_embedded);
                
                i++;
            }
            
            // Check if we generated "<USER>"
            if (strncmp(potential_end_tag, "<USER>", 6) == 0) {
                printf("\n\n[End of generation - detected new user turn]\n");
                break;
            }
        }
        
        // Update for next token
        unsigned char current_byte = next_byte;
        CHECK_CUDA(cudaMemcpy(d_input_byte, &current_byte, sizeof(unsigned char), cudaMemcpyHostToDevice));
        embeddings_forward(embeddings, d_input_byte, d_input_embedded, 1);
        forward_pass(ssm, d_input_embedded);
    }
    
    printf("\n\n[Generation complete - reached token limit]\n");
    
    // Clean up
    free(h_output_probs);
    free_embeddings(embeddings);
    free_ssm(ssm);
    cudaFree(d_input_byte);
    cudaFree(d_input_embedded);
    
    if (model_filename && argc < 2)
        free(model_filename);
    if (embedding_filename && argc < 3)
        free(embedding_filename);
    
    return 0;
}