#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Get the total size of a file
size_t get_file_size(const char* filename);

// Create shuffled sequence indices for entire corpus
size_t* create_shuffled_indices(size_t total_sequences);

// Sample sequences using shuffled indices
size_t sample_sequences(const char* filename, size_t* indices, size_t start_idx, int seq_len, unsigned char* input_tokens, unsigned char* target_tokens, size_t num_sequences);

#endif