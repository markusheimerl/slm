#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "../data.h"
#include "../bpe/bpe.h"
#include "gpt.h"

GPT* gpt = NULL;
Tokenizer* tokenizer = NULL;

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

// Generate text autoregressively from a prompt
void generate_text(GPT* gpt, Tokenizer* tok, float temperature, const char* prompt, int gen_len) {
    // Encode prompt
    GArray* prompt_tokens = encode_tokenizer(tok, prompt);
    
    // Start with zero-initialized sequence
    unsigned short* tokens = (unsigned short*)calloc(gpt->seq_len, sizeof(unsigned short));
    
    // Copy prompt tokens
    for (guint i = 0; i < prompt_tokens->len && i < (guint)gpt->seq_len; i++) {
        tokens[i] = (unsigned short)g_array_index(prompt_tokens, uint32_t, i);
    }
    
    int start_pos = (int)prompt_tokens->len;
    g_array_free(prompt_tokens, TRUE);
    
    printf("\"%s", prompt);
    fflush(stdout);
    
    // Allocate device memory for single sequence
    unsigned short *d_tokens;
    CHECK_CUDA(cudaMalloc(&d_tokens, gpt->seq_len * sizeof(unsigned short)));
    float* h_logits = (float*)malloc(gpt->vocab_size * sizeof(float));
    
    // Generate tokens one at a time
    for (int pos = start_pos - 1; pos < gen_len && pos + 1 < gpt->seq_len; pos++) {
        // Copy current sequence to device
        CHECK_CUDA(cudaMemcpy(d_tokens, tokens, gpt->seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
        
        // Forward pass to get logits (batch_size=1 for generation)
        GPT* gen_gpt = gpt;  // Use existing model but with batch_size context
        int saved_batch = gen_gpt->batch_size;
        gen_gpt->batch_size = 1;  // Temporarily set to 1 for generation
        forward_pass_gpt(gen_gpt, d_tokens);
        gen_gpt->batch_size = saved_batch;  // Restore
        
        // Copy logits for current position back to host
        CHECK_CUDA(cudaMemcpy(h_logits, &gen_gpt->d_output[pos * gen_gpt->vocab_size], 
                              gen_gpt->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature scaling and find max for numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < gen_gpt->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int v = 0; v < gen_gpt->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        for (int v = 0; v < gen_gpt->vocab_size; v++) {
            h_logits[v] /= sum_exp;
        }
        
        // Sample from the distribution
        float r = (float)rand() / (float)RAND_MAX;
        unsigned short next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < gen_gpt->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }
        
        // Add sampled token to sequence
        tokens[pos + 1] = next_token;
        
        // Decode and print just this token
        GArray* single_token = g_array_new(FALSE, FALSE, sizeof(uint32_t));
        uint32_t tok_val = (uint32_t)next_token;
        g_array_append_val(single_token, tok_val);
        gchar* decoded = decode_tokenizer(tok, single_token);
        printf("%s", decoded);
        fflush(stdout);
        g_free(decoded);
        g_array_free(single_token, TRUE);
    }
    
    printf("\"\n");
    CHECK_CUDA(cudaFree(d_tokens));
    free(h_logits);
    free(tokens);
}

// Tokenize corpus and save to binary file
void tokenize_corpus(Tokenizer* tok, const char* input_file, const char* output_file) {
    printf("Tokenizing corpus...\n");
    
    FILE* in = fopen(input_file, "r");
    if (!in) {
        g_error("Cannot open input file: %s", input_file);
    }
    
    FILE* out = fopen(output_file, "wb");
    if (!out) {
        fclose(in);
        g_error("Cannot open output file: %s", output_file);
    }
    
    char* line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    size_t total_tokens = 0;
    size_t line_count = 0;
    
    while ((line_len = getline(&line, &line_cap, in)) != -1) {
        if (line_len > 0 && line[line_len-1] == '\n') {
            line[line_len-1] = '\0';
            line_len--;
        }
        
        if (line_len == 0) continue;
        
        GArray* tokens = encode_tokenizer(tok, line);
        
        // Write tokens as unsigned short
        for (guint i = 0; i < tokens->len; i++) {
            uint32_t token = g_array_index(tokens, uint32_t, i);
            unsigned short token_short = (unsigned short)token;
            fwrite(&token_short, sizeof(unsigned short), 1, out);
        }
        
        total_tokens += tokens->len;
        line_count++;
        
        if (line_count % 10000 == 0) {
            printf("  Processed %zu lines, %zu tokens\n", line_count, total_tokens);
        }
        
        g_array_free(tokens, TRUE);
    }
    
    free(line);
    fclose(in);
    fclose(out);
    
    printf("Tokenization complete: %zu lines, %zu tokens\n", line_count, total_tokens);
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
    
    // Initialize tokenizer
    const char* tokenizer_file = "../bpe/tokenizer.bin";
    const char* corpus_file = "../corpus.txt";
    const char* tokenized_corpus = "tokenized_corpus.bin";
    
    tokenizer = init_tokenizer();
    
    // Load or train tokenizer
    FILE* tok_check = fopen(tokenizer_file, "rb");
    if (tok_check) {
        fclose(tok_check);
        load_tokenizer(tokenizer, tokenizer_file);
    } else {
        printf("Training tokenizer...\n");
        train_tokenizer(tokenizer, corpus_file, 65536, 1ULL * 1024 * 1024 * 1024);
        save_tokenizer(tokenizer, tokenizer_file);
    }
    
    // Get vocab size from tokenizer
    int vocab_size = 256 + g_hash_table_size(tokenizer->merges);
    printf("Vocabulary size: %d\n", vocab_size);
    
    // Tokenize corpus if needed
    FILE* corpus_check = fopen(tokenized_corpus, "rb");
    if (!corpus_check) {
        tokenize_corpus(tokenizer, corpus_file, tokenized_corpus);
    } else {
        fclose(corpus_check);
        printf("Using existing tokenized corpus\n");
    }
    
    // Initialize or load model
    if (argc > 1) {
        gpt = load_gpt(argv[1], batch_size, seq_len, cublaslt_handle);
    } else {
        gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size, vocab_size, cublaslt_handle);
    }
    
    printf("Parameters: ~%.1fM\n", (float)(vocab_size * d_model + d_model * vocab_size + num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Create shuffled indices for random sampling without replacement
    size_t total_tokens = get_file_size(tokenized_corpus) / sizeof(unsigned short);
    size_t total_sequences = (total_tokens - 1) / seq_len;
    size_t* shuffled_indices = create_shuffled_indices(total_sequences);
    
    // Allocate host buffers for sequences
    size_t sequences_per_chunk = (128 * 1024 * 1024) / (seq_len * sizeof(unsigned short));
    unsigned short* input_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    unsigned short* target_tokens = (unsigned short*)malloc(sequences_per_chunk * seq_len * sizeof(unsigned short));
    
    // Allocate device buffers
    unsigned short *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(unsigned short)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(unsigned short)));
    
    // Training loop: process corpus in chunks with random sampling
    for (size_t chunk_idx = 0; chunk_idx < total_sequences / sequences_per_chunk; chunk_idx++) {
        // Sample next chunk of sequences from shuffled corpus
        sample_sequences(tokenized_corpus, &shuffled_indices[chunk_idx * sequences_per_chunk], seq_len, input_tokens, target_tokens, sequences_per_chunk);
        
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
                   loss / log(2.0) / 2.0,
                   ((double)total_sequences / batch_size - (chunk_idx * (sequences_per_chunk / batch_size) + batch) - 1) * ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / 3600.0);
        }
        
        // Generate sample text
        printf("\n--- Sample ---\n");
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>The capital of France is", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>The chemical symbol of gold is", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>If yesterday was Friday, then tomorrow will be", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>The opposite of hot is", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>The planets of the solar system are:", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>My favorite color is", 64);
        generate_text(gpt, tokenizer, 0.7f, "<|bos|>If 5*x + 3 = 13, then x is", 64);
        printf("--- End ---\n\n");
        
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
    free_tokenizer(tokenizer);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}