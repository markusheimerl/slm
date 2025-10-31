#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Get the total size of a file
size_t get_file_size(const char* filename);

// Read a chunk from an open file
size_t read_chunk(FILE* f, char* buffer, size_t size);

// Generate random training sequences from a corpus chunk
void generate_sequences(unsigned char* input_tokens, unsigned char* target_tokens, int num_sequences, int seq_len, char* chunk, size_t chunk_size);

// Calculate total number of batches we'll train on
size_t calculate_total_batches(const char* filename, int seq_len, int batch_size, size_t chunk_size);

#endif