#include "data.h"

size_t get_corpus_size(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return 0;
    
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fclose(file);
    return size;
}

char* load_corpus_chunk(const char* filename, size_t offset, size_t chunk_size, size_t* loaded_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    
    if (offset >= file_size) {
        fclose(file);
        return NULL;
    }
    
    size_t to_read = chunk_size;
    if (offset + to_read > file_size) {
        to_read = file_size - offset;
    }
    
    char* buffer = (char*)malloc(to_read + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    
    fseek(file, offset, SEEK_SET);
    *loaded_size = fread(buffer, 1, to_read, file);
    buffer[*loaded_size] = '\0';
    
    fclose(file);
    return buffer;
}

void generate_sequences(unsigned char* input_tokens, unsigned char* target_tokens, int num_sequences, int seq_len, char* corpus, size_t corpus_size) {
    if (corpus_size < (size_t)seq_len) return;
    
    size_t max_start = corpus_size - seq_len;
    
    for (int i = 0; i < num_sequences; i++) {
        size_t start = rand() % (max_start + 1);
        
        for (int j = 0; j < seq_len; j++) {
            input_tokens[i * seq_len + j] = (unsigned char)corpus[start + j];
        }
        
        for (int j = 0; j < seq_len - 1; j++) {
            target_tokens[i * seq_len + j] = (unsigned char)corpus[start + j + 1];
        }
        
        if (start + seq_len < corpus_size) {
            target_tokens[i * seq_len + seq_len - 1] = (unsigned char)corpus[start + seq_len];
        } else {
            target_tokens[i * seq_len + seq_len - 1] = (unsigned char)' ';
        }
    }
}