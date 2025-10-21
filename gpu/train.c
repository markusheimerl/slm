#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include "../data.h"
#include "slm.h"

// Constants
const char* BPE_TOKENIZER_PATH = "../bpe/gpu/20251019_162333_bpe.bin";
const char* CORPUS_PATH = "../corpus.txt";
const size_t CHUNK_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB
const char* TEMP_CHECKPOINT = "temp_chunk_checkpoint.bin";

// Global state
SLM* slm = NULL;
time_t training_start_time;
struct timespec last_accumulation_time;

// SIGINT handler to save model and exit
void handle_sigint(int signum) {
    if (slm) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
        save_slm(slm, model_filename, BPE_TOKENIZER_PATH);
    }
    exit(128 + signum);
}

// Format seconds into hh:mm:ss
void format_time(int total_seconds, char* buffer, size_t buffer_size) {
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;
    snprintf(buffer, buffer_size, "%02d:%02d:%02d", hours, minutes, seconds);
}

// Get current time in milliseconds
long long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

// Get file size
size_t get_file_size(const char* filename) {
    struct stat st;
    if (stat(filename, &st) != 0) {
        printf("Error: Could not stat file %s\n", filename);
        return 0;
    }
    return st.st_size;
}

// Load a chunk of the corpus
char* load_corpus_chunk(const char* filename, size_t offset, size_t chunk_size, size_t* bytes_read) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    if (fseek(file, offset, SEEK_SET) != 0) {
        printf("Error: Could not seek to offset %zu\n", offset);
        fclose(file);
        return NULL;
    }
    
    char* buffer = (char*)malloc(chunk_size + 1);
    if (!buffer) {
        printf("Error: Could not allocate memory for chunk\n");
        fclose(file);
        return NULL;
    }
    
    *bytes_read = fread(buffer, 1, chunk_size, file);
    buffer[*bytes_read] = '\0';
    
    fclose(file);
    return buffer;
}

// Split tokenized corpus into sequences
void tokenize_to_sequences(uint32_t* corpus_tokens, uint32_t total_tokens, 
                          uint32_t** input_tokens, uint32_t** target_tokens, 
                          int* num_sequences, int seq_len) {
    *num_sequences = (total_tokens - 1) / seq_len;
    
    *input_tokens = (uint32_t*)malloc(*num_sequences * seq_len * sizeof(uint32_t));
    *target_tokens = (uint32_t*)malloc(*num_sequences * seq_len * sizeof(uint32_t));
    
    for (int seq = 0; seq < *num_sequences; seq++) {
        size_t start_pos = seq * seq_len;
        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            (*input_tokens)[idx] = corpus_tokens[start_pos + t];
            (*target_tokens)[idx] = corpus_tokens[start_pos + t + 1];
        }
    }
}

// Generate text from random corpus sample
void generate_from_corpus(SLM* slm, uint32_t* input_tokens, int num_sequences, int length, 
                         float temperature, uint32_t* d_input_tokens) {
    // Pick a random sequence as seed
    int random_seq = rand() % num_sequences;
    uint32_t* seed_tokens = &input_tokens[random_seq * slm->seq_len];
    
    // Copy seed to host buffer
    uint32_t* h_seed_tokens = (uint32_t*)malloc(slm->seq_len * sizeof(uint32_t));
    for (int i = 0; i < slm->seq_len; i++) h_seed_tokens[i] = seed_tokens[i];
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Display seed
    printf("Seed: \"");
    char* seed_text = decode_bpe(slm->bpe, h_seed_tokens, slm->seq_len);
    printf("%s", seed_text);
    free(seed_text);
    printf("\" -> \"");
    fflush(stdout);
    
    // Allocate logits buffer
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    
    // Generate text one token at a time
    for (int gen = 0; gen < length; gen++) {
        forward_pass_slm(slm, d_input_tokens);
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[(slm->seq_len - 1) * slm->vocab_size], 
                             slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            float exp_val = expf(h_logits[v] - max_logit);
            h_logits[v] = exp_val;
            sum_exp += exp_val;
        }
        for (int v = 0; v < slm->vocab_size; v++) h_logits[v] /= sum_exp;
        
        // Sample
        float r = (float)rand() / (float)RAND_MAX;
        uint32_t next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }

        // Display
        char* token_text = decode_bpe(slm->bpe, &next_token, 1);
        printf("%s", token_text);
        free(token_text);
        fflush(stdout);
        
        // Shift sequence left and add new token
        for (int i = 0; i < slm->seq_len - 1; i++) h_seed_tokens[i] = h_seed_tokens[i + 1];
        h_seed_tokens[slm->seq_len - 1] = next_token;
        
        // Update device memory
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    
    printf("\"\n");
    free(h_seed_tokens);
    free(h_logits);
}

// Generate text from a prompt
void generate_from_prompt(SLM* slm, const char* prompt, int length, float temperature, uint32_t* d_input_tokens) {
    size_t prompt_len = strlen(prompt);
    uint32_t num_prompt_tokens;
    uint32_t* prompt_tokens = encode_bpe(slm->bpe, prompt, prompt_len, &num_prompt_tokens);
    
    if (num_prompt_tokens >= (uint32_t)slm->seq_len) {
        printf("Error: prompt too long (%u tokens, max %d)\n", num_prompt_tokens, slm->seq_len);
        free(prompt_tokens);
        return;
    }
    
    uint32_t* h_seed_tokens = (uint32_t*)malloc(slm->seq_len * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_prompt_tokens; i++) h_seed_tokens[i] = prompt_tokens[i];
    for (int i = num_prompt_tokens; i < slm->seq_len; i++) h_seed_tokens[i] = 0;
    
    CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    printf("%s", prompt);
    fflush(stdout);
    
    float* h_logits = (float*)malloc(slm->vocab_size * sizeof(float));
    int current_pos = num_prompt_tokens - 1;
    
    for (int gen = 0; gen < length; gen++) {
        forward_pass_slm(slm, d_input_tokens);
        CHECK_CUDA(cudaMemcpy(h_logits, &slm->output_mlp->d_output[current_pos * slm->vocab_size], 
                             slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Apply temperature
        float max_logit = -1e30f;
        for (int v = 0; v < slm->vocab_size; v++) {
            h_logits[v] /= temperature;
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            float exp_val = expf(h_logits[v] - max_logit);
            h_logits[v] = exp_val;
            sum_exp += exp_val;
        }
        for (int v = 0; v < slm->vocab_size; v++) h_logits[v] /= sum_exp;
        
        // Sample
        float r = (float)rand() / (float)RAND_MAX;
        uint32_t next_token = 0;
        float cumsum = 0.0f;
        for (int v = 0; v < slm->vocab_size; v++) {
            cumsum += h_logits[v];
            if (r <= cumsum) {
                next_token = v;
                break;
            }
        }

        // Display
        char* token_text = decode_bpe(slm->bpe, &next_token, 1);
        printf("%s", token_text);
        free(token_text);
        fflush(stdout);
        
        // Update sequence
        current_pos++;
        if (current_pos >= slm->seq_len) {
            for (int i = 0; i < slm->seq_len - 1; i++) h_seed_tokens[i] = h_seed_tokens[i + 1];
            h_seed_tokens[slm->seq_len - 1] = next_token;
            current_pos = slm->seq_len - 1;
        } else {
            h_seed_tokens[current_pos] = next_token;
        }
        
        CHECK_CUDA(cudaMemcpy(d_input_tokens, h_seed_tokens, slm->seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    
    printf("\n");
    free(prompt_tokens);
    free(h_seed_tokens);
    free(h_logits);
}

// Load and tokenize a chunk - returns 0 on success, -1 on failure
int load_and_tokenize_chunk(BPE* bpe, size_t corpus_offset, size_t total_corpus_size, int seq_len,
                            uint32_t** input_tokens, uint32_t** target_tokens, int* num_sequences,
                            size_t* chunk_bytes_out) {
    size_t remaining = total_corpus_size - corpus_offset;
    size_t this_chunk_size = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
    
    size_t chunk_bytes;
    char* chunk = load_corpus_chunk(CORPUS_PATH, corpus_offset, this_chunk_size, &chunk_bytes);
    if (!chunk) return -1;
    
    printf("Loaded %.2f GB\n", (double)chunk_bytes / (1024.0 * 1024.0 * 1024.0));
    
    uint32_t num_tokens;
    uint32_t* tokens = encode_bpe(bpe, chunk, chunk_bytes, &num_tokens);
    free(chunk);
    
    printf("Tokenized: %u tokens\n", num_tokens);
    
    tokenize_to_sequences(tokens, num_tokens, input_tokens, target_tokens, num_sequences, seq_len);
    free(tokens);
    
    printf("Created %d sequences\n", *num_sequences);
    
    *chunk_bytes_out = chunk_bytes;
    return 0;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    // Initialize cuBLAS
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Load BPE tokenizer
    printf("Loading BPE tokenizer from %s\n", BPE_TOKENIZER_PATH);
    BPE* bpe = load_bpe(BPE_TOKENIZER_PATH);
    if (!bpe) {
        printf("Failed to load BPE tokenizer\n");
        return 1;
    }
    printf("BPE vocab size: %u\n", bpe->vocab_size);

    // Model hyperparameters
    const int seq_len = 2048;
    const int d_model = 1024;
    const int hidden_dim = 4096;
    const int num_layers = 32;
    const int batch_size = 2;
    const int accumulation_steps = 1;
    const float learning_rate = 0.00001f;
    
    // Get corpus size
    size_t total_corpus_size = get_file_size(CORPUS_PATH);
    if (total_corpus_size == 0) {
        printf("Failed to get corpus size\n");
        return 1;
    }
    printf("Total corpus size: %.2f GB\n", (double)total_corpus_size / (1024.0 * 1024.0 * 1024.0));
    
    // Load first chunk
    printf("\n=== Loading First Chunk ===\n");
    uint32_t* input_tokens = NULL;
    uint32_t* target_tokens = NULL;
    int num_sequences;
    size_t first_chunk_bytes;
    
    if (load_and_tokenize_chunk(bpe, 0, total_corpus_size, seq_len, 
                                &input_tokens, &target_tokens, &num_sequences, &first_chunk_bytes) != 0) {
        printf("Failed to load first chunk\n");
        return 1;
    }
    
    // Estimate total batches using first chunk data
    printf("\n=== Estimating Total Batches ===\n");
    int num_batches = num_sequences / batch_size;
    double bytes_per_batch = (double)first_chunk_bytes / (double)num_batches;
    int estimated_total_batches = (int)((double)total_corpus_size / bytes_per_batch);
    
    printf("First chunk: %zu bytes, %d sequences, %d batches\n", first_chunk_bytes, num_sequences, num_batches);
    printf("Bytes per batch: %.2f\n", bytes_per_batch);
    printf("Estimated total batches: %d\n", estimated_total_batches);
    
    // Initialize or load model
    if (argc > 1) {
        printf("\nLoading checkpoint: %s\n", argv[1]);
        slm = load_slm(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("\nInitializing new model...\n");
        slm = init_slm(bpe, seq_len, d_model, hidden_dim, num_layers, batch_size, cublaslt_handle);
    }
    
    printf("Total parameters: ~%.1fM\n", 
           (float)(slm->vocab_size * d_model + d_model * slm->vocab_size + 
                   num_layers * (4 * d_model * d_model + d_model * hidden_dim + hidden_dim * d_model)) / 1e6f);
    
    // Test prompts
    const char* prompts[] = {
        "<|bos|>The capital of France is ",
        "<|bos|>The chemical symbol of gold is ",
        "<|bos|>If yesterday was Friday, then tomorrow will be ",
        "<|bos|>The opposite of hot is ",
        "<|bos|>The planets of the solar system are: ",
        "<|bos|>My favorite color is ",
        "<|bos|>If 5*x + 3 = 13, then x is "
    };
    const int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    
    // Allocate device memory
    uint32_t *d_input_tokens, *d_target_tokens;
    CHECK_CUDA(cudaMalloc(&d_input_tokens, batch_size * seq_len * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_target_tokens, batch_size * seq_len * sizeof(uint32_t)));
    
    // Training loop
    printf("\n=== Starting Training ===\n");
    training_start_time = time(NULL);
    size_t corpus_offset = first_chunk_bytes;
    int global_batch_counter = 0;
    int chunk_number = 1;
    
    // Initialize timing
    clock_gettime(CLOCK_MONOTONIC, &last_accumulation_time);
    
    while (true) {
        printf("\n=== Chunk %d ===\n", chunk_number);
        
        int num_batches_this_chunk = num_sequences / batch_size;
        
        for (int batch = 0; batch < num_batches_this_chunk; batch++) {
            global_batch_counter++;
            
            // Copy batch to device
            int batch_offset = batch * batch_size * seq_len;
            CHECK_CUDA(cudaMemcpy(d_input_tokens, &input_tokens[batch_offset], 
                                 batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_tokens, &target_tokens[batch_offset], 
                                 batch_size * seq_len * sizeof(uint32_t), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_slm(slm, d_input_tokens);
            float loss = calculate_loss_slm(slm, d_target_tokens);
            if(loss >= 10.0) raise(SIGINT);

            // Backward pass with gradient accumulation
            if (batch % accumulation_steps == 0) zero_gradients_slm(slm);
            backward_pass_slm(slm, d_input_tokens);
            if ((batch + 1) % accumulation_steps == 0) {
                update_weights_slm(slm, learning_rate, batch_size * accumulation_steps);
            }
            
            // Print progress with ETA and DT
            if (batch % accumulation_steps == 0) {
                // Calculate ETA
                time_t now = time(NULL);
                int elapsed_seconds = (int)difftime(now, training_start_time);
                double avg_seconds_per_batch = (double)elapsed_seconds / (double)global_batch_counter;
                int remaining_batches = estimated_total_batches - global_batch_counter;
                int eta_seconds = (int)(avg_seconds_per_batch * remaining_batches);
                
                char eta_str[16];
                format_time(eta_seconds, eta_str, sizeof(eta_str));
                
                // Calculate DT (delta time in milliseconds)
                long long current_time_ms = get_time_ms();
                struct timespec current_time;
                clock_gettime(CLOCK_MONOTONIC, &current_time);
                long long last_time_ms = (long long)last_accumulation_time.tv_sec * 1000LL + 
                                         last_accumulation_time.tv_nsec / 1000000LL;
                long long dt_ms = current_time_ms - last_time_ms;
                last_accumulation_time = current_time;
                
                printf("Batch [%d/%d] (Chunk %d, Local %d/%d), Loss: %.6f, ETA: %s, DT: %lldms\n", 
                       global_batch_counter, estimated_total_batches, chunk_number, batch, 
                       num_batches_this_chunk, loss, eta_str, dt_ms);
            }
            
            // Generate samples periodically
            if (global_batch_counter % 200 == 0) {
                printf("\n--- Generated sample (batch %d/%d) ---\n", global_batch_counter, estimated_total_batches);
                generate_from_corpus(slm, input_tokens, num_sequences, 128, 0.8f, d_input_tokens);
                printf("\n");
                for (int p = 0; p < num_prompts; p++) {
                    generate_from_prompt(slm, prompts[p], 64, 0.01f, d_input_tokens);
                }
                printf("--- End sample ---\n\n");
                
                // Reset timing after generation
                clock_gettime(CLOCK_MONOTONIC, &last_accumulation_time);
            }

            // Checkpoint periodically
            if (global_batch_counter % 1000 == 0) {
                save_slm(slm, "checkpoint_slm.bin", BPE_TOKENIZER_PATH);
            }
        }
        
        // Check if we've processed the entire corpus
        if (corpus_offset >= total_corpus_size) break;
        
        // Prepare for next chunk
        chunk_number++;
        
        printf("\nSaving temporary checkpoint...\n");
        save_slm(slm, TEMP_CHECKPOINT, BPE_TOKENIZER_PATH);
        
        printf("Freeing model from GPU...\n");
        free_slm(slm);
        slm = NULL;
        
        printf("Freeing previous chunk data...\n");
        free(input_tokens);
        free(target_tokens);
        
        // Load next chunk
        printf("\n=== Loading Next Chunk ===\n");
        size_t chunk_bytes;
        if (load_and_tokenize_chunk(bpe, corpus_offset, total_corpus_size, seq_len,
                                   &input_tokens, &target_tokens, &num_sequences, &chunk_bytes) != 0) {
            printf("Failed to load chunk at offset %zu\n", corpus_offset);
            break;
        }
        corpus_offset += chunk_bytes;
        
        printf("Reloading model...\n");
        slm = load_slm(TEMP_CHECKPOINT, batch_size, cublaslt_handle);
        
        // Reset timing after chunk loading
        clock_gettime(CLOCK_MONOTONIC, &last_accumulation_time);
    }

    // Training complete
    printf("\n=== Training Complete ===\n");
    printf("Total batches processed: %d\n", global_batch_counter);
    
    time_t now = time(NULL);
    int total_time = (int)difftime(now, training_start_time);
    char total_time_str[16];
    format_time(total_time, total_time_str, sizeof(total_time_str));
    printf("Total training time: %s\n", total_time_str);

    // Save final model
    char model_fname[64];
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    save_slm(slm, model_fname, BPE_TOKENIZER_PATH);
    
    // Cleanup
    free(input_tokens);
    free(target_tokens);
    CHECK_CUDA(cudaFree(d_input_tokens));
    CHECK_CUDA(cudaFree(d_target_tokens));
    free_slm(slm);
    free_bpe(bpe);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}