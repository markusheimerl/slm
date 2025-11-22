#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "../data.h"
#include "gpt.h"

GPT* gpt = NULL;

// Pre-tokenized samples from encode_samples
unsigned short sample_0[] = {440, 659, 439, 456, 3389, 281, 4351, 308}; // "<|bos|>The capital of France is"
unsigned short sample_1[] = {440, 659, 439, 456, 2739, 3775, 281, 3864, 308}; // "<|bos|>The chemical symbol of gold is"
unsigned short sample_2[] = {440, 659, 439, 1611, 15740, 427, 9282, 44, 934, 14093, 512, 307}; // "<|bos|>If yesterday was Friday, then tomorrow will be"
unsigned short sample_3[] = {440, 659, 439, 456, 6228, 281, 3251, 308}; // "<|bos|>The opposite of hot is"
unsigned short sample_4[] = {440, 659, 439, 456, 8694, 281, 261, 3218, 800, 355, 58}; // "<|bos|>The planets of the solar system are:"
unsigned short sample_5[] = {440, 659, 439, 5936, 6969, 2640, 308}; // "<|bos|>My favorite color is"
unsigned short sample_6[] = {440, 659, 439, 1611, 32, 53, 49203, 1110, 32, 51, 686, 32, 1633, 44, 934, 1601, 308}; // "<|bos|>If 5*x + 3 = 13, then x is"

int sample_lengths[] = {8, 9, 12, 8, 11, 7, 17};

// Signal handler to save model on Ctrl+C
void handle_sigint(int signum) {
    if (gpt) {
        char filename[64];
        time_t now = time(NULL);
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_gpt.bin", localtime(&now));
        save_gpt(gpt, filename);
    }
    exit(128 + signum);
}

// Generate text autoregressively from pre-tokenized prompt and save to file
void generate_and_save_tokens(GPT* gpt, float temperature, unsigned short* d_input_tokens, unsigned short* prompt_tokens, int prompt_len, int gen_len, const char* output_file) {
    // Start with zero-initialized sequence
    unsigned short* h_tokens = (unsigned short*)calloc(gpt->seq_len, sizeof(unsigned short));
    
    // Copy prompt tokens
    for (int i = 0; i < prompt_len; i++) {
        h_tokens[i] = prompt_tokens[i];
    }
    
    float* h_logits = (float*)malloc(gpt->vocab_size * sizeof(float));
    
    // Generate tokens one at a time
    for (int pos = prompt_len - 1; pos < gen_len; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, gpt->seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
        
        // Forward pass to get logits
        forward_pass_gpt(gpt, d_input_tokens);
        
        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits, &gpt->d_output[pos * gpt->vocab_size], gpt->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned short next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Add sampled token to sequence
        h_tokens[pos + 1] = next_token;
    }
    
    // Save the complete sequence to file (append mode)
    FILE* f = fopen(output_file, "ab");
    if (f) {
        // Write the complete sequence (prompt + generated tokens)
        fwrite(h_tokens, sizeof(unsigned short), gen_len + 1, f);
        fclose(f);
    }
    
    free(h_tokens);
    free(h_logits);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLAS
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Model hyperparameters
    const int seq_len = 512;
    const int num_layers = 16;
    const int batch_size = 30;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    const float learning_rate = 0.00004f;
    
    // Initialize or load model
    if (argc > 1) {
        gpt = load_gpt(argv[1], batch_size, seq_len, cublaslt_handle);
    } else {
        gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(gpt->vocab_size * d_model + d_model * gpt->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Create shuffled indices for random sampling without replacement
    size_t total_sequences = (get_file_size("../bpe/corpus_tokenized.bin") - 2) / (2 * seq_len);
    size_t* shuffled_indices = create_shuffled_indices(total_sequences);
    
    // Allocate host buffers for sequences
    size_t sequences_per_chunk = (128 * 1024 * 1024) / (seq_len * 2);
    unsigned short* input_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    unsigned short* target_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    
    // Allocate device buffers
    unsigned short *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned short)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned short)));
    
    // Array of sample pointers for easy iteration
    unsigned short* samples[] = {sample_0, sample_1, sample_2, sample_3, sample_4, sample_5, sample_6};
    int num_samples = 7;
    
    // Training loop: process corpus in chunks with random sampling
    for (size_t chunk_idx = 0; chunk_idx < total_sequences / sequences_per_chunk; chunk_idx++) {
        // Sample next chunk of sequences from shuffled corpus
        sample_sequences("../bpe/corpus_tokenized.bin", &shuffled_indices[chunk_idx * sequences_per_chunk], seq_len, input_tokens, target_tokens, sequences_per_chunk);
        
        // Train on all batches in this chunk
        for (int batch = 0; batch < (int)(sequences_per_chunk / batch_size); batch++) {
            struct timespec start; clock_gettime(CLOCK_MONOTONIC, &start);
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch * batch_size * seq_len], batch_size * seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_gpt(gpt, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_gpt(gpt, d_target_tokens);
            if (loss >= 12.0) raise(SIGINT);
            
            // Backward pass
            zero_gradients_gpt(gpt);
            backward_pass_gpt(gpt, d_input_tokens);
            
            // Update weights with cosine learning rate schedule
            float lr = learning_rate * fminf(1.0f, (float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / 100.0f) * (0.5f * (1.0f + cosf(M_PI * ((float)(chunk_idx * (sequences_per_chunk / batch_size) + batch) / (float)(total_sequences / batch_size)))));
            update_weights_gpt(gpt, lr, batch_size);
            
            CHECK_CUDA(cudaDeviceSynchronize()); struct timespec end; clock_gettime(CLOCK_MONOTONIC, &end);
            printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f, dt: %.2fms, tok/s: %.0f, bpb: %.4f, ETA: %.1fh\n",
                   chunk_idx, total_sequences / sequences_per_chunk, batch, (int)(sequences_per_chunk / batch_size),
                   loss, lr, ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6),
                   (batch_size * seq_len) / ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9),
                   loss / log(2.0) / 4.8,
                   ((double)total_sequences / batch_size - (chunk_idx * (sequences_per_chunk / batch_size) + batch) - 1) * ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / 3600.0);
        }
        
        // Generate sample sequences and save tokens to file
        printf("\n--- Generating samples ---\n");
        for (int i = 0; i < num_samples; i++) {
            generate_and_save_tokens(gpt, 0.001f, d_input_tokens, samples[i], sample_lengths[i], 32, "generated_samples_tokenized.bin");
            printf("Sample %d generated\n", i);
        }
        printf("--- Samples saved to generated_samples_tokenized.bin ---\n\n");
        
        // Save checkpoint
        save_gpt(gpt, "checkpoint_gpt.bin");
    }
    
    // Save final model with timestamp
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_gpt.bin", localtime(&now));
    save_gpt(gpt, filename);
    
    // Cleanup
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free(shuffled_indices);
    free_gpt(gpt);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}