#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "gpt.h"

// Encode text as byte-pairs
int encode_text(const char* text, unsigned short* tokens, int max_tokens) {
    int len = strlen(text);
    int num_tokens = 0;
    
    for (int i = 0; i < len && num_tokens < max_tokens; i += 2) {
        if (i + 1 < len) {
            tokens[num_tokens++] = (unsigned short)((unsigned char)text[i] << 8) | (unsigned char)text[i + 1];
        } else {
            tokens[num_tokens++] = (unsigned short)((unsigned char)text[i] << 8) | (unsigned char)' ';
        }
    }
    
    return num_tokens;
}

// Calculate average negative log-likelihood for continuation given context
float calculate_continuation_loss(GPT* gpt, unsigned short* tokens, int context_len, int total_len) {
    if (total_len <= context_len || total_len > gpt->seq_len) return 1e30f;
    
    // Allocate and prepare input
    unsigned short* h_input = (unsigned short*)calloc(gpt->seq_len, sizeof(unsigned short));
    memcpy(h_input, tokens, total_len * sizeof(unsigned short));
    
    unsigned short* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, gpt->seq_len * sizeof(unsigned short)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, gpt->seq_len * sizeof(unsigned short), cudaMemcpyHostToDevice));
    
    // Forward pass
    forward_pass_gpt(gpt, d_input);
    
    // Calculate loss for continuation tokens only
    float total_loss = 0.0f;
    int continuation_len = total_len - context_len;
    float* h_logits = (float*)malloc(gpt->vocab_size * sizeof(float));
    
    for (int i = context_len; i < total_len; i++) {
        // Get logits for position i-1 to predict token i
        CHECK_CUDA(cudaMemcpy(h_logits, &gpt->d_output[(i - 1) * gpt->vocab_size], 
                              gpt->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Softmax with numerical stability
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            if (h_logits[v] > max_logit) max_logit = h_logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        
        // Cross-entropy for target token
        unsigned short target = tokens[i];
        float prob = h_logits[target] / sum_exp;
        total_loss += -logf(prob + 1e-10f);
    }
    
    free(h_logits);
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    
    return total_loss / continuation_len;
}

// Simple JSON parsing
void parse_json_string(const char* json, const char* key, char* value, int max_len) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\": \"", key);
    const char* start = strstr(json, search);
    if (!start) { value[0] = '\0'; return; }
    start += strlen(search);
    const char* end = strchr(start, '"');
    if (!end) { value[0] = '\0'; return; }
    int len = end - start;
    if (len >= max_len) len = max_len - 1;
    strncpy(value, start, len);
    value[len] = '\0';
}

int parse_json_int(const char* json, const char* key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\": ", key);
    const char* start = strstr(json, search);
    return start ? atoi(start + strlen(search)) : -1;
}

int parse_json_array(const char* json, const char* key, char endings[][2048], int max_count) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\": [", key);
    const char* start = strstr(json, search);
    if (!start) return 0;
    start += strlen(search);
    
    int count = 0;
    const char* pos = start;
    while (count < max_count && (pos = strchr(pos, '"'))) {
        pos++;
        const char* end = pos;
        while (*end && *end != '"') {
            if (*end == '\\' && *(end + 1)) end++;
            end++;
        }
        if (!*end) break;
        
        int len = end - pos;
        if (len >= 2047) len = 2047;
        strncpy(endings[count], pos, len);
        endings[count][len] = '\0';
        count++;
        
        pos = end + 1;
        while (*pos && isspace(*pos)) pos++;
        if (*pos == ']') break;
    }
    
    return count;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    printf("Loading model from %s...\n", argv[1]);
    GPT* gpt = load_gpt(argv[1], 1, cublaslt_handle);
    if (!gpt) {
        printf("Failed to load model\n");
        CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
        return 1;
    }
    printf("Model loaded (seq_len=%d, d_model=%d, layers=%d)\n", 
           gpt->seq_len, gpt->d_model, gpt->num_layers);
    
    FILE* f = fopen("../hellaswag_val.jsonl", "r");
    if (!f) {
        printf("Error: cannot open ../hellaswag_val.jsonl\n");
        free_gpt(gpt);
        CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
        return 1;
    }
    
    int total = 0, correct = 0;
    char line[32768];
    
    printf("\nEvaluating HellaSwag validation set...\n\n");
    
    while (fgets(line, sizeof(line), f)) {
        char ctx[4096], endings[4][2048];
        int label = parse_json_int(line, "label");
        parse_json_string(line, "ctx", ctx, sizeof(ctx));
        int num_endings = parse_json_array(line, "endings", endings, 4);
        
        if (label < 0 || num_endings != 4 || strlen(ctx) == 0) continue;
        
        unsigned short ctx_tokens[2048];
        int ctx_len = encode_text(ctx, ctx_tokens, 2048);
        
        float losses[4];
        int best = 0;
        float best_loss = 1e30f;
        
        for (int i = 0; i < 4; i++) {
            char combined[8192];
            snprintf(combined, sizeof(combined), "%s%s", ctx, endings[i]);
            unsigned short tokens[4096];
            int total_len = encode_text(combined, tokens, 4096);
            
            losses[i] = calculate_continuation_loss(gpt, tokens, ctx_len, total_len);
            if (losses[i] < best_loss) {
                best_loss = losses[i];
                best = i;
            }
        }
        
        total++;
        if (best == label) correct++;
        
        printf("Progress: %d examples, accuracy: %.2f%%\n", total, 100.0f * correct / total);
    }
    
    fclose(f);
    
    printf("\n=== HellaSwag Results ===\n");
    printf("Total: %d\n", total);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %.2f%%\n", 100.0f * correct / total);
    
    free_gpt(gpt);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}