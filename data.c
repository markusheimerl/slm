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

// Read a chunk from an open file
size_t read_chunk(FILE* f, char* buffer, size_t size) {
    return fread(buffer, 1, size, f);
}

// Generate random training sequences from a corpus chunk
void generate_sequences(unsigned char* input_tokens, unsigned char* target_tokens, int seq_len, char* chunk, size_t chunk_size) {
    int num_sequences = chunk_size / seq_len;

    for (int i = 0; i < num_sequences; i++) {
        // Pick a random starting position in the chunk
        size_t start = rand() % (chunk_size - seq_len);
        
        // Copy input sequence and target sequence (shifted by 1)
        for (int j = 0; j < seq_len; j++) {
            input_tokens[i * seq_len + j] = chunk[start + j];
            target_tokens[i * seq_len + j] = chunk[start + j + 1];
        }
    }
}

// Calculate total number of batches we'll train on
size_t calculate_total_batches(const char* filename, int seq_len, int batch_size, size_t chunk_size) {
    size_t total_size = get_file_size(filename);
    size_t num_complete_chunks = total_size / chunk_size;
    size_t sequences_per_chunk = chunk_size / seq_len;
    size_t batches_per_chunk = sequences_per_chunk / batch_size;
    return num_complete_chunks * batches_per_chunk;
}

// Calculate current batch number
size_t calculate_batch_number(FILE* f, size_t chunk_size, int current_batch_in_chunk, int seq_len, int batch_size) {
    size_t chunks_completed = (ftell(f) / chunk_size) - 1;
    size_t batches_per_chunk = (chunk_size / seq_len) / batch_size;
    return chunks_completed * batches_per_chunk + current_batch_in_chunk + 1;
}