#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include "ssm/gpu/ssm.h"

// ---------------------------------------------------------------------
// Structure for holding embeddings
// ---------------------------------------------------------------------
typedef struct {
    // Embeddings (device and host memory)
    float* d_embeddings;      // vocab_size x embedding_dim
    float* h_embeddings;      // vocab_size x embedding_dim
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
} Embeddings;

// ---------------------------------------------------------------------
// CUDA kernel: Embed input bytes (forward pass)
// ---------------------------------------------------------------------
__global__ void embed_bytes_kernel(float* output, 
                                  const unsigned char* bytes, 
                                  const float* embeddings, 
                                  int batch_size, 
                                  int embedding_dim) {
    int batch_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (batch_idx < batch_size && emb_idx < embedding_dim) {
        // Get the byte value for this batch item
        unsigned char byte_val = bytes[batch_idx];
        
        // Calculate position in embedding table
        int embedding_offset = byte_val * embedding_dim;
        
        // Copy the embedding vector to output
        output[batch_idx * embedding_dim + emb_idx] = embeddings[embedding_offset + emb_idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Softmax for probabilities output
// ---------------------------------------------------------------------
__global__ void softmax_kernel(float* logits, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointer to this batch item's prediction vector
        float* batch_logits = logits + batch_idx * vocab_size;
        
        // Find max value for numerical stability
        float max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
        
        // Compute exp(logits - max) and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] = expf(batch_logits[i] - max_val);
            sum_exp += batch_logits[i];
        }
        
        // Ensure sum is not zero
        sum_exp = fmaxf(sum_exp, 1e-10f);
        
        // Normalize to get probabilities
        for (int i = 0; i < vocab_size; i++) {
            batch_logits[i] /= sum_exp;
        }
    }
}

// ---------------------------------------------------------------------
// Function: Load embeddings
// Loads the embeddings from a binary file
// ---------------------------------------------------------------------
Embeddings* load_embeddings(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    Embeddings* emb = (Embeddings*)malloc(sizeof(Embeddings));
    
    // Read dimensions
    fread(&emb->vocab_size, sizeof(int), 1, file);
    fread(&emb->embedding_dim, sizeof(int), 1, file);
    
    printf("Loading embeddings: vocab_size=%d, embedding_dim=%d\n", 
           emb->vocab_size, emb->embedding_dim);
    
    // Allocate host memory for embeddings
    emb->h_embeddings = (float*)malloc(emb->vocab_size * emb->embedding_dim * sizeof(float));
    
    // Read embeddings from file to host memory
    fread(emb->h_embeddings, sizeof(float), emb->vocab_size * emb->embedding_dim, file);
    
    // Allocate and copy to device memory
    CHECK_CUDA(cudaMalloc(&emb->d_embeddings, emb->vocab_size * emb->embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(emb->d_embeddings, emb->h_embeddings, 
                         emb->vocab_size * emb->embedding_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Embeddings loaded from %s\n", filename);
    return emb;
}

// ---------------------------------------------------------------------
// Function: Forward pass for embeddings
// ---------------------------------------------------------------------
void embeddings_forward(Embeddings* emb, unsigned char* d_bytes, float* d_output, int batch_size) {
    embed_bytes_kernel<<<batch_size, emb->embedding_dim>>>(
        d_output, d_bytes, emb->d_embeddings, batch_size, emb->embedding_dim);
}

// ---------------------------------------------------------------------
// Function: Free embeddings
// ---------------------------------------------------------------------
void free_embeddings(Embeddings* emb) {
    if (!emb) return;
    
    cudaFree(emb->d_embeddings);
    free(emb->h_embeddings);
    free(emb);
}

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
    
    // Default model filenames (can be overridden by command line args)
    char* model_filename = NULL;
    char* embedding_filename = NULL;
    
    // Default sampling parameters
    float temperature = 0.8f;
    float top_p = 0.9f;
    int max_tokens = 512;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_filename = argv[++i];
        } else if (strcmp(argv[i], "--embeddings") == 0 && i + 1 < argc) {
            embedding_filename = argv[++i];
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        }
    }
    
    // Find the most recent model and embeddings files if not specified
    if (!model_filename || !embedding_filename) {
        FILE* fp = popen("ls -t *_slm.bin | head -1", "r");
        if (fp) {
            char buffer[256];
            if (fgets(buffer, sizeof(buffer), fp) != NULL) {
                size_t len = strlen(buffer);
                if (len > 0 && buffer[len-1] == '\n')
                    buffer[len-1] = '\0';
                
                if (!model_filename) {
                    model_filename = strdup(buffer);
                }
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
                
                if (!embedding_filename) {
                    embedding_filename = strdup(buffer);
                }
            }
            pclose(fp);
        }
    }
    
    if (!model_filename || !embedding_filename) {
        fprintf(stderr, "Error: Could not find model and/or embeddings files\n");
        return EXIT_FAILURE;
    }
    
    printf("=== SLM Generation Parameters ===\n");
    printf("Model file: %s\n", model_filename);
    printf("Embeddings file: %s\n", embedding_filename);
    printf("Temperature: %.2f\n", temperature);
    printf("Top-p: %.2f\n", top_p);
    printf("Max tokens: %d\n\n", max_tokens);
    
    // Load embeddings and model
    Embeddings* embeddings = load_embeddings(embedding_filename);
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
    
    if (model_filename && !strstr(model_filename, "_slm.bin"))
        free(model_filename);
    if (embedding_filename && !strstr(embedding_filename, "_embeddings.bin"))
        free(embedding_filename);
    
    return 0;
}