#include "data.h"

// Get the total size of a file
size_t get_file_size(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fclose(f);
    return size;
}

// Create shuffled sequence indices for entire corpus
size_t* create_shuffled_indices(size_t total_sequences) {
    size_t* indices = (size_t*)malloc(total_sequences * sizeof(size_t));
    
    // Initialize sequentially
    for (size_t i = 0; i < total_sequences; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle
    for (size_t i = total_sequences - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    return indices;
}

// Sample sequences using shuffled indices
void sample_sequences(const char* filename, size_t* indices, int seq_len, unsigned short* input_tokens, unsigned short* target_tokens, size_t num_sequences) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    
    unsigned char* buffer = (unsigned char*)malloc((seq_len + 1) * 2 * sizeof(unsigned char));
    
    for (size_t i = 0; i < num_sequences; i++) {
        fseek(f, indices[i] * seq_len * 2, SEEK_SET);
        
        if (fread(buffer, 1, (seq_len + 1) * 2, f) < (size_t)((seq_len + 1) * 2)) break;
        
        for (int j = 0; j < seq_len; j++) {
            input_tokens[i * seq_len + j] = (unsigned short)((buffer[j * 2] << 8) | buffer[j * 2 + 1]);
            target_tokens[i * seq_len + j] = (unsigned short)((buffer[(j + 1) * 2] << 8) | buffer[(j + 1) * 2 + 1]);
        }
    }
    
    free(buffer);
    fclose(f);
}