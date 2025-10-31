#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <ctype.h>
#include "../data.h"
#include "slm.h"

// wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl

SLM* slm = NULL;

// Signal handler to save model on Ctrl+C
void handle_sigint(int signum) {
    if (slm) {
        char filename[64];
        time_t now = time(NULL);
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
        save_slm(slm, filename);
    }
    exit(128 + signum);
}

// Generate text autoregressively from a prompt
void generate_text(SLM* slm, float temperature, unsigned char* d_input_tokens, const char* bos, int gen_len) {
    // Start with zero-initialized sequence
    unsigned char* h_tokens = (unsigned char*)calloc(slm->seq_len, sizeof(unsigned char));
    
    // Set beginning of sequence (prompt)
    for (int i = 0; i < (int)strlen(bos); i++) {
        h_tokens[i] = (unsigned char)bos[i];
    }
    
    printf("\"%s", bos);
    fflush(stdout);
    
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate characters one at a time
    for (int pos = strlen(bos) - 1; pos < gen_len; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        // Forward pass to get logits
        forward_pass_slm(slm, d_input_tokens);
        
        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[pos * slm->vocab_size], slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned char next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Add sampled token to sequence
        h_tokens[pos + 1] = next_token;
        printf("%c", (char)next_token);
        fflush(stdout);
    }
    
    printf("\"\n");
    free(h_tokens);
    free(h_logits);
}

// Simplified HellaSwag evaluation with extensive debug printing
float evaluate_hellaswag_simple(SLM* slm, const char* filename, unsigned char* d_tokens, int max_examples) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Warning: Could not open HellaSwag file: %s\n", filename);
        return -1.0f;
    }
    
    printf("\n========================================\n");
    printf("Starting HellaSwag Evaluation\n");
    printf("========================================\n\n");
    
    int correct = 0;
    int total = 0;
    
    unsigned char* h_tokens = (unsigned char*)malloc(slm->seq_len * sizeof(unsigned char));
    float* h_logits = (float*)malloc(slm->seq_len * slm->vocab_size * sizeof(float));
    
    char line[8192];
    while (fgets(line, sizeof(line), f) && total < max_examples) {
        // Parse the line manually - simple arrays, no structs
        char context[2048] = {0};
        char endings[4][2048] = {0};
        int correct_label = -1;
        
        // Extract context
        char* ctx_start = strstr(line, "\"ctx\": \"");
        if (!ctx_start) ctx_start = strstr(line, "\"ctx\":\"");
        if (ctx_start) {
            ctx_start = strchr(ctx_start, ':') + 1;
            while (*ctx_start == ' ' || *ctx_start == '"') ctx_start++;
            
            char* ctx_end = ctx_start;
            while (*ctx_end && !(*ctx_end == '"' && *(ctx_end - 1) != '\\')) ctx_end++;
            
            int ctx_len = ctx_end - ctx_start;
            if (ctx_len >= 2048) ctx_len = 2047;
            strncpy(context, ctx_start, ctx_len);
            context[ctx_len] = '\0';
            
            // Unescape special characters
            char* src = context;
            char* dst = context;
            while (*src) {
                if (*src == '\\' && *(src + 1) == 'n') { *dst++ = ' '; src += 2; }
                else if (*src == '\\' && *(src + 1) == '"') { *dst++ = '"'; src += 2; }
                else if (*src == '\\' && *(src + 1) == '\\') { *dst++ = '\\'; src += 2; }
                else { *dst++ = *src++; }
            }
            *dst = '\0';
        }
        
        // Extract label
        char* label_start = strstr(line, "\"label\": ");
        if (!label_start) label_start = strstr(line, "\"label\":");
        if (label_start) {
            label_start = strchr(label_start, ':') + 1;
            while (*label_start == ' ') label_start++;
            correct_label = atoi(label_start);
        }
        
        // Extract endings
        char* endings_start = strstr(line, "\"endings\": [");
        if (!endings_start) endings_start = strstr(line, "\"endings\":[");
        if (endings_start) {
            endings_start = strchr(endings_start, '[') + 1;
            
            for (int i = 0; i < 4; i++) {
                char* ending_start = strchr(endings_start, '"');
                if (!ending_start) break;
                ending_start++;
                
                char* ending_end = ending_start;
                while (*ending_end) {
                    if (*ending_end == '"' && *(ending_end - 1) != '\\') break;
                    ending_end++;
                }
                
                int ending_len = ending_end - ending_start;
                if (ending_len >= 2048) ending_len = 2047;
                strncpy(endings[i], ending_start, ending_len);
                endings[i][ending_len] = '\0';
                
                // Unescape
                char* src = endings[i];
                char* dst = endings[i];
                while (*src) {
                    if (*src == '\\' && *(src + 1) == 'n') { *dst++ = ' '; src += 2; }
                    else if (*src == '\\' && *(src + 1) == '"') { *dst++ = '"'; src += 2; }
                    else if (*src == '\\' && *(src + 1) == '\\') { *dst++ = '\\'; src += 2; }
                    else { *dst++ = *src++; }
                }
                *dst = '\0';
                
                endings_start = ending_end + 1;
            }
        }
        
        // Skip if parsing failed
        if (context[0] == '\0' || correct_label < 0) {
            printf("Warning: Failed to parse line, skipping\n");
            continue;
        }
        
        // Print example info
        printf("========================================\n");
        printf("Example %d:\n", total + 1);
        printf("----------------------------------------\n");
        printf("Context: \"%s\"\n\n", context);
        
        float best_score = -1e30f;
        int best_ending = -1;
        
        // Evaluate each ending
        for (int e = 0; e < 4; e++) {
            printf("  Ending %d: \"%s\"\n", e, endings[e]);
            
            // Construct full text: context + ending
            char full_text[4096];
            snprintf(full_text, sizeof(full_text), "%s%s", context, endings[e]);
            
            int text_len = strlen(full_text);
            if (text_len > slm->seq_len) {
                printf("    [!] Text too long (%d chars), truncating to %d\n", text_len, slm->seq_len);
                text_len = slm->seq_len;
            }
            
            int ctx_len = strlen(context);
            printf("    [i] Context: %d chars, Full text: %d chars, Ending: %d chars\n", 
                   ctx_len, text_len, text_len - ctx_len);
            
            // Convert to tokens (just byte values)
            memset(h_tokens, 0, slm->seq_len);
            for (int j = 0; j < text_len; j++) {
                h_tokens[j] = (unsigned char)full_text[j];
            }
            
            printf("    [i] First 20 tokens: ");
            for (int j = 0; j < 20 && j < text_len; j++) {
                printf("%d ", h_tokens[j]);
            }
            printf("\n");
            
            // Run model
            CHECK_CUDA(cudaMemcpy(d_tokens, h_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            forward_pass_slm(slm, d_tokens);
            CHECK_CUDA(cudaMemcpy(h_logits, slm->output_mlp->d_output, slm->seq_len * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Compute score for ending tokens only (not context)
            if (ctx_len >= text_len - 1) {
                printf("    [!] No ending tokens to evaluate (context fills entire text)!\n\n");
                continue;
            }
            
            float total_log_prob = 0.0f;
            int count = 0;
            
            printf("    [i] Computing log probabilities for ending tokens (positions %d to %d):\n", 
                   ctx_len, text_len - 2);
            
            // Show first 5 token predictions in detail
            for (int t = ctx_len; t < text_len - 1; t++) {
                float* logits_t = &h_logits[t * slm->vocab_size];
                
                // Find max logit for numerical stability
                float max_logit = -1e30f;
                for (int v = 0; v < slm->vocab_size; v++) {
                    if (logits_t[v] > max_logit) max_logit = logits_t[v];
                }
                
                // Compute log softmax denominator
                float sum_exp = 0.0f;
                for (int v = 0; v < slm->vocab_size; v++) {
                    sum_exp += expf(logits_t[v] - max_logit);
                }
                float log_sum_exp = logf(sum_exp);
                
                // Get log probability of the actual next token
                unsigned char next_token = h_tokens[t + 1];
                float log_prob = (logits_t[next_token] - max_logit) - log_sum_exp;
                float prob = expf(log_prob);
                
                total_log_prob += log_prob;
                count++;
                
                // Show detail for first 5 tokens
                if (count <= 5) {
                    char display_char = (next_token >= 32 && next_token < 127) ? next_token : '?';
                    printf("        Token %d: '%c' (byte %3d) -> logit=%.2f, prob=%.4f, log_prob=%.4f\n", 
                           count, display_char, next_token, logits_t[next_token], prob, log_prob);
                }
            }
            
            if (count > 5) {
                printf("        ... (%d more tokens)\n", count - 5);
            }
            
            float avg_log_prob = (count > 0) ? total_log_prob / count : -1e30f;
            
            printf("    [i] Evaluated %d tokens\n", count);
            printf("    [i] Total log prob: %.6f\n", total_log_prob);
            printf("    [i] Average log prob: %.6f\n", avg_log_prob);
            printf("    [i] Perplexity: %.2f\n", expf(-avg_log_prob));
            
            if (avg_log_prob > best_score) {
                best_score = avg_log_prob;
                best_ending = e;
                printf("    >>> NEW BEST SCORE <<<\n");
            }
            printf("\n");
        }
        
        printf("  Model's choice: Ending %d (avg log prob: %.6f)\n", best_ending, best_score);
        printf("  Correct answer: Ending %d\n", correct_label);
        
        if (best_ending == correct_label) {
            correct++;
            printf("  ✓✓✓ CORRECT ✓✓✓\n");
        } else {
            printf("  ✗✗✗ WRONG ✗✗✗\n");
        }
        
        printf("\n  Running accuracy: %.2f%% (%d correct out of %d)\n", 
               100.0f * correct / (total + 1), correct, total + 1);
        printf("========================================\n\n");
        
        total++;
    }
    
    fclose(f);
    free(h_tokens);
    free(h_logits);
    
    if (total == 0) {
        printf("ERROR: No examples were evaluated!\n");
        return -1.0f;
    }
    
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║       HELLASWAG FINAL RESULTS          ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("  Total examples:  %d\n", total);
    printf("  Correct:         %d\n", correct);
    printf("  Wrong:           %d\n", total - correct);
    printf("  Accuracy:        %.2f%%\n", 100.0f * correct / total);
    printf("  Random baseline: 25.00%%\n");
    printf("  Improvement:     %.2f%%\n", (100.0f * correct / total) - 25.0f);
    printf("════════════════════════════════════════\n\n");
    
    return (float)correct / total;
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
    const int batch_size = 24;
    const int d_model = num_layers * 64;
    const int hidden_dim = d_model * 4;
    const float learning_rate = 0.00003f;
    
    // Initialize or load model
    if (argc > 1) {
        slm = load_slm(argv[1], batch_size, cublaslt_handle);
    } else {
        slm = init_slm(seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Open corpus file
    FILE* f = fopen("../corpus.txt", "rb");
    
    // Calculate total chunks
    size_t chunk_size = 1 * 1024 * 1024;
    size_t total_chunks = get_file_size("../corpus.txt") / chunk_size;
    
    // Allocate host buffers
    char* chunk = (char*)malloc(chunk_size);
    int max_sequences = chunk_size / seq_len;
    unsigned char* input_tokens = (unsigned char*)malloc(max_sequences * seq_len);
    unsigned char* target_tokens = (unsigned char*)malloc(max_sequences * seq_len);
    
    // Allocate device buffers
    unsigned char *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len));
    
    // Training loop: process corpus in chunks
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
        // Read next chunk
        size_t loaded = read_chunk(f, chunk, chunk_size);
        if (loaded < chunk_size) break;
        
        // Generate random training sequences from chunk
        generate_sequences(input_tokens, target_tokens, seq_len, chunk, loaded);
        
        // Calculate batches in this chunk
        int batches_in_chunk = ((int)loaded / seq_len) / batch_size;
        
        // Train on all batches in this chunk
        for (int batch = 0; batch < batches_in_chunk; batch++) {
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch * batch_size * seq_len], batch_size * seq_len, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch * batch_size * seq_len], batch_size * seq_len, cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            
            // Calculate loss
            float loss = calculate_loss_slm(slm, d_target_tokens);
            if (loss >= 7.0) raise(SIGINT);
            
            // Backward pass
            zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            
            // Update weights with cosine learning rate schedule
            float lr = learning_rate * (0.5f * (1.0f + cosf(M_PI * ((float)((calculate_batch_number(f, chunk_size, batch, seq_len, batch_size)) - 1) / (float)(calculate_total_batches("../corpus.txt", seq_len, batch_size, chunk_size))))));
            update_weights_slm(slm, lr, batch_size);
            
            printf("Chunk [%zu/%zu], Batch [%d/%d], Loss: %.6f, LR: %.7f\n", chunk_idx, total_chunks, batch, batches_in_chunk, loss, lr);
        }
        
        // Generate sample text
        printf("\n--- Sample ---\n");
        generate_text(slm, 0.9f, d_input_tokens, "The opposite of hot is ", 32);
        printf("--- End ---\n\n");
        
        // Evaluate on HellaSwag
        evaluate_hellaswag_simple(slm, "../hellaswag_val.jsonl", d_input_tokens, 16);
        
        // Save checkpoint
        save_slm(slm, "checkpoint_slm.bin");
    }
    
    // Save final model with timestamp
    char filename[64];
    time_t now = time(NULL);
    strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_slm(slm, filename);
    
    // Cleanup
    fclose(f);
    free(chunk);
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_slm(slm);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}