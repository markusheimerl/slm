#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <ctype.h>
#include "../data.h"
#include "slm.h"

// wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl
/*
markus@tower:~/Desktop/slm$ head ./hellaswag_val.jsonl
{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}
{"ind": 92, "activity_label": "Clean and jerk", "ctx_a": "A lady walks to a barbell. She bends down and grabs the pole.", "ctx_b": "the lady", "ctx": "A lady walks to a barbell. She bends down and grabs the pole. the lady", "split": "val", "split_type": "zeroshot", "label": 3, "endings": ["swings and lands in her arms.", "pulls the barbell forward.", "pulls a rope attached to the barbell.", "stands and lifts the weight over her head."], "source_id": "activitynet~v_-lJS58hyo1c"}
{"ind": 106, "activity_label": "Canoeing", "ctx_a": "Two women in a child are shown in a canoe while a man pulls the canoe while standing in the water, with other individuals visible in the background.", "ctx_b": "the child and a different man", "ctx": "Two women in a child are shown in a canoe while a man pulls the canoe while standing in the water, with other individuals visible in the background. the child and a different man", "split": "val", "split_type": "indomain", "label": 2, "endings": ["are then shown paddling down a river in a boat while a woman talks.", "are driving the canoe, they go down the river flowing side to side.", "sit in a canoe while the man paddles.", "walking go down the rapids, while the man in his helicopter almost falls and goes out of canoehood."], "source_id": "activitynet~v_-xQvJmC2jhk"}
{"ind": 114, "activity_label": "High jump", "ctx_a": "A boy is running down a track.", "ctx_b": "the boy", "ctx": "A boy is running down a track. the boy", "split": "val", "split_type": "zeroshot", "label": 2, "endings": ["runs into a car.", "gets in a mat.", "lifts his body above the height of a pole.", "stands on his hands and springs."], "source_id": "activitynet~v_-zHX3Gdx6I4"}
{"ind": 116, "activity_label": "High jump", "ctx_a": "The boy lifts his body above the height of a pole. The boy lands on his back on to a red mat.", "ctx_b": "the boy", "ctx": "The boy lifts his body above the height of a pole. The boy lands on his back on to a red mat. the boy", "split": "val", "split_type": "zeroshot", "label": 1, "endings": ["turns his body around on the mat.", "gets up from the mat.", "continues to lift his body over the pole.", "wiggles out of the mat."], "source_id": "activitynet~v_-zHX3Gdx6I4"}
{"ind": 117, "activity_label": "High jump", "ctx_a": "The boy lands on his back on to a red mat. The boy gets up from the mat.", "ctx_b": "the boy", "ctx": "The boy lands on his back on to a red mat. The boy gets up from the mat. the boy", "split": "val", "split_type": "zeroshot", "label": 1, "endings": ["starts doing spins.", "celebrates by clapping and flexing both arms.", "is dancing on the mat.", "does jump jacks on his stick."], "source_id": "activitynet~v_-zHX3Gdx6I4"}
{"ind": 149, "activity_label": "Playing harmonica", "ctx_a": "A man is standing in front of a camera. He starts playing a harmonica for the camera.", "ctx_b": "he", "ctx": "A man is standing in front of a camera. He starts playing a harmonica for the camera. he", "split": "val", "split_type": "zeroshot", "label": 2, "endings": ["begins to play the harmonica with his body while looking at the camera.", "seems to be singing while playing the harmonica.", "rocks back and forth to the music as he goes.", "painted a fence in front of the camera."], "source_id": "activitynet~v_0RUMAGGab1k"}
{"ind": 170, "activity_label": "Sumo", "ctx_a": "A cartoon animation video is shown with people wandering around and rockets being shot.", "ctx_b": "two men", "ctx": "A cartoon animation video is shown with people wandering around and rockets being shot. two men", "split": "val", "split_type": "indomain", "label": 0, "endings": ["fight robots of evil and ends with a to be continued.", "are then shown in closeups shooting a shot put.", "push a child in a speedboat in the water.", "look in the cameraman's eye and smile."], "source_id": "activitynet~v_0WVkoTBmhA0"}
{"ind": 180, "activity_label": "Sharpening knives", "ctx_a": "A man is holding a pocket knife while sitting on some rocks in the wilderness.", "ctx_b": "then he", "ctx": "A man is holding a pocket knife while sitting on some rocks in the wilderness. then he", "split": "val", "split_type": "zeroshot", "label": 1, "endings": ["opens a can of oil put oil on the knife, and puts oil on a knife and press it through a can filled with oil then cuts several pieces from the sandwiches.", "takes a small stone from the flowing river and smashes it on another stone.", "uses the knife to shave his leg.", "sand the rocks and tops them by using strong pressure."], "source_id": "activitynet~v_0bosp4-pyTM"}
{"ind": 182, "activity_label": "Sharpening knives", "ctx_a": "Then he takes a small stone from the flowing river and smashes it on another stone. He starts to crush the small stone to smaller pieces.", "ctx_b": "he", "ctx": "Then he takes a small stone from the flowing river and smashes it on another stone. He starts to crush the small stone to smaller pieces. he", "split": "val", "split_type": "zeroshot", "label": 1, "endings": ["cuts the center stone in half and blow it on to make it bigger.", "grind it hard to make the pieces smaller.", "eventually brings it back into view and adds it to the smaller ones to make a small triangular shaped piece.", "starts to party with them and throw the pieces by hand while they celebrate."], "source_id": "activitynet~v_0bosp4-pyTM"}
markus@tower:~/Desktop/slm$ 
*/

// HellaSwag structures and functions
#define MAX_HELLASWAG_EXAMPLES 1000
#define MAX_TEXT_LENGTH 2048
#define NUM_ENDINGS 4

typedef struct {
    char ctx[MAX_TEXT_LENGTH];
    char endings[NUM_ENDINGS][MAX_TEXT_LENGTH];
    int label;
} HellaSwagExample;

typedef struct {
    HellaSwagExample* examples;
    int num_examples;
} HellaSwagDataset;

// Load HellaSwag validation dataset
HellaSwagDataset* load_hellaswag(const char* filename, int max_examples) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Warning: Could not open HellaSwag file: %s\n", filename);
        return NULL;
    }
    
    HellaSwagDataset* dataset = (HellaSwagDataset*)malloc(sizeof(HellaSwagDataset));
    dataset->examples = (HellaSwagExample*)malloc(max_examples * sizeof(HellaSwagExample));
    dataset->num_examples = 0;
    
    char line[8192];
    while (fgets(line, sizeof(line), f) && dataset->num_examples < max_examples) {
        HellaSwagExample* ex = &dataset->examples[dataset->num_examples];
        
        // Parse JSON manually
        char* ctx_start = strstr(line, "\"ctx\": \"");
        if (!ctx_start) ctx_start = strstr(line, "\"ctx\":\"");
        
        char* label_start = strstr(line, "\"label\": ");
        if (!label_start) label_start = strstr(line, "\"label\":");
        
        char* endings_start = strstr(line, "\"endings\": [");
        if (!endings_start) endings_start = strstr(line, "\"endings\":[");
        
        if (!ctx_start || !label_start || !endings_start) {
            continue;
        }
        
        // Extract context - skip past "ctx": " or "ctx":"
        ctx_start = strchr(ctx_start, ':') + 1;
        while (*ctx_start == ' ' || *ctx_start == '"') ctx_start++;
        
        char* ctx_end = ctx_start;
        while (*ctx_end && !(*ctx_end == '"' && *(ctx_end - 1) != '\\')) ctx_end++;
        
        int ctx_len = ctx_end - ctx_start;
        if (ctx_len >= MAX_TEXT_LENGTH) ctx_len = MAX_TEXT_LENGTH - 1;
        strncpy(ex->ctx, ctx_start, ctx_len);
        ex->ctx[ctx_len] = '\0';
        
        // Unescape special characters
        char* src = ex->ctx;
        char* dst = ex->ctx;
        while (*src) {
            if (*src == '\\' && *(src + 1) == 'n') { *dst++ = ' '; src += 2; }
            else if (*src == '\\' && *(src + 1) == '"') { *dst++ = '"'; src += 2; }
            else if (*src == '\\' && *(src + 1) == '\\') { *dst++ = '\\'; src += 2; }
            else { *dst++ = *src++; }
        }
        *dst = '\0';
        
        // Extract label - it's an integer after "label":
        label_start = strchr(label_start, ':') + 1;
        while (*label_start == ' ') label_start++;
        ex->label = atoi(label_start);
        
        // Extract endings - skip past "endings": [ or "endings":[
        endings_start = strchr(endings_start, '[') + 1;
        
        for (int i = 0; i < NUM_ENDINGS; i++) {
            // Find opening quote
            char* ending_start = strchr(endings_start, '"');
            if (!ending_start) break;
            ending_start++;
            
            // Find closing quote (not escaped)
            char* ending_end = ending_start;
            while (*ending_end) {
                if (*ending_end == '"' && *(ending_end - 1) != '\\') break;
                ending_end++;
            }
            
            int ending_len = ending_end - ending_start;
            if (ending_len >= MAX_TEXT_LENGTH) ending_len = MAX_TEXT_LENGTH - 1;
            strncpy(ex->endings[i], ending_start, ending_len);
            ex->endings[i][ending_len] = '\0';
            
            // Unescape
            src = ex->endings[i];
            dst = ex->endings[i];
            while (*src) {
                if (*src == '\\' && *(src + 1) == 'n') { *dst++ = ' '; src += 2; }
                else if (*src == '\\' && *(src + 1) == '"') { *dst++ = '"'; src += 2; }
                else if (*src == '\\' && *(src + 1) == '\\') { *dst++ = '\\'; src += 2; }
                else { *dst++ = *src++; }
            }
            *dst = '\0';
            
            endings_start = ending_end + 1;
        }
        
        dataset->num_examples++;
    }
    
    fclose(f);
    
    if (dataset->num_examples > 0) {
        printf("Loaded %d HellaSwag examples from %s\n", dataset->num_examples, filename);
    } else {
        printf("Warning: Failed to parse any examples from %s\n", filename);
        free(dataset->examples);
        free(dataset);
        return NULL;
    }
    
    return dataset;
}

void free_hellaswag(HellaSwagDataset* dataset) {
    if (dataset) {
        free(dataset->examples);
        free(dataset);
    }
}

// Evaluate HellaSwag
float evaluate_hellaswag(SLM* slm, HellaSwagDataset* dataset, unsigned char* d_tokens) {
    if (!dataset) return -1.0f;
    
    int correct = 0;
    unsigned char* h_tokens = (unsigned char*)malloc(slm->seq_len * sizeof(unsigned char));
    float* h_logits = (float*)malloc(slm->seq_len * slm->vocab_size * sizeof(float));
    
    printf("  Evaluating");
    fflush(stdout);
    
    for (int i = 0; i < dataset->num_examples; i++) {
        if (i % 20 == 0) { printf("."); fflush(stdout); }
        
        HellaSwagExample* ex = &dataset->examples[i];
        
        float best_score = -1e30f;
        int best_ending = -1;
        
        // Evaluate each ending
        for (int e = 0; e < NUM_ENDINGS; e++) {
            // Construct full text: context + ending
            char full_text[MAX_TEXT_LENGTH * 2];
            snprintf(full_text, sizeof(full_text), "%s%s", ex->ctx, ex->endings[e]);
            
            int text_len = strlen(full_text);
            if (text_len > slm->seq_len) text_len = slm->seq_len;
            
            // Convert to tokens
            memset(h_tokens, 0, slm->seq_len);
            for (int j = 0; j < text_len; j++) {
                h_tokens[j] = (unsigned char)full_text[j];
            }
            
            // Copy to device and forward pass
            CHECK_CUDA(cudaMemcpy(d_tokens, h_tokens, slm->seq_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
            forward_pass_slm(slm, d_tokens);
            
            // Copy logits back
            CHECK_CUDA(cudaMemcpy(h_logits, slm->output_mlp->d_output, slm->seq_len * slm->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Compute score for the ending part only
            int ctx_len = strlen(ex->ctx);
            if (ctx_len >= text_len - 1) continue;
            
            float total_log_prob = 0.0f;
            int count = 0;
            
            for (int t = ctx_len; t < text_len - 1; t++) {
                float* logits_t = &h_logits[t * slm->vocab_size];
                
                // Find max for numerical stability
                float max_logit = -1e30f;
                for (int v = 0; v < slm->vocab_size; v++) {
                    if (logits_t[v] > max_logit) max_logit = logits_t[v];
                }
                
                // Compute log softmax
                float sum_exp = 0.0f;
                for (int v = 0; v < slm->vocab_size; v++) {
                    sum_exp += expf(logits_t[v] - max_logit);
                }
                float log_sum_exp = logf(sum_exp);
                
                // Get log probability of next token
                unsigned char next_token = h_tokens[t + 1];
                float log_prob = (logits_t[next_token] - max_logit) - log_sum_exp;
                total_log_prob += log_prob;
                count++;
            }
            
            float score = (count > 0) ? total_log_prob / count : -1e30f;
            
            if (score > best_score) {
                best_score = score;
                best_ending = e;
            }
        }
        
        if (best_ending == ex->label) {
            correct++;
        }
    }
    
    printf(" done!\n");
    
    free(h_tokens);
    free(h_logits);
    
    return (float)correct / dataset->num_examples;
}

SLM* slm = NULL;
HellaSwagDataset* hellaswag_val = NULL;

// Signal handler to save model on Ctrl+C
void handle_sigint(int signum) {
    if (slm) {
        char filename[64];
        time_t now = time(NULL);
        strftime(filename, sizeof(filename), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
        save_slm(slm, filename);
    }
    if (hellaswag_val) free_hellaswag(hellaswag_val);
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
    
    // Load HellaSwag validation set
    hellaswag_val = load_hellaswag("../hellaswag_val.jsonl", 200);
    if (!hellaswag_val) {
        printf("Continuing without HellaSwag evaluation...\n");
    }
    
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
        generate_text(slm, 0.9f, d_input_tokens, "The opposite of hot is ", 8);
        printf("--- End ---\n\n");
        
        // Evaluate on HellaSwag
        if (hellaswag_val) {
            printf("HellaSwag Evaluation (%d examples):\n", hellaswag_val->num_examples);
            float accuracy = evaluate_hellaswag(slm, hellaswag_val, d_input_tokens);
            printf("  Accuracy: %.2f%% (Random baseline: 25.00%%)\n\n", accuracy * 100.0f);
        }
        
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
    if (hellaswag_val) free_hellaswag(hellaswag_val);
    
    return 0;
}