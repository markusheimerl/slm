#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Load text data with memory streaming
void load_text_data(const char* filename, int** d_input_ids, int** d_target_ids, 
                   int* num_batches, int seq_len, int batch_size, int max_batches) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s\n", filename);
        exit(1);
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Calculate available sequences
    int max_sequences = (file_size - 1) / seq_len;
    int available_batches = max_sequences / batch_size;
    
    *num_batches = (max_batches > 0 && max_batches < available_batches) ? 
                   max_batches : available_batches;
    
    int total_sequences = (*num_batches) * batch_size;
    
    printf("File size: %ld bytes, using %d batches of %d sequences\n", 
           file_size, *num_batches, batch_size);
    
    // Allocate host memory
    int* h_input_ids = (int*)malloc(total_sequences * seq_len * sizeof(int));
    int* h_target_ids = (int*)malloc(total_sequences * seq_len * sizeof(int));
    
    // Load data
    for (int seq = 0; seq < total_sequences; seq++) {
        unsigned char buffer[seq_len + 1];
        size_t bytes_read = fread(buffer, 1, seq_len + 1, file);
        
        if (bytes_read < (size_t)(seq_len + 1)) {
            fprintf(stderr, "Warning: Not enough data for sequence %d\n", seq);
            break;
        }
        
        for (int pos = 0; pos < seq_len; pos++) {
            h_input_ids[seq * seq_len + pos] = (int)buffer[pos];
            h_target_ids[seq * seq_len + pos] = (int)buffer[pos + 1];
        }
    }
    
    fclose(file);
    
    // Allocate device memory and copy data
    cudaMalloc(d_input_ids, total_sequences * seq_len * sizeof(int));
    cudaMalloc(d_target_ids, total_sequences * seq_len * sizeof(int));
    cudaMemcpy(*d_input_ids, h_input_ids, total_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_target_ids, h_target_ids, total_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice);
    
    free(h_input_ids);
    free(h_target_ids);
}

// Reshape data from [batch][time] to [time][batch]
void reshape_data_for_batch_processing(int* input_ids, int* target_ids,
                                     int** input_reshaped, int** target_reshaped,
                                     int num_sequences, int seq_len) {
    *input_reshaped = (int*)malloc(seq_len * num_sequences * sizeof(int));
    *target_reshaped = (int*)malloc(seq_len * num_sequences * sizeof(int));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time]
            int orig_idx = b * seq_len + t;
            
            // New layout: [time][seq] 
            int new_idx = t * num_sequences + b;
            
            (*input_reshaped)[new_idx] = input_ids[orig_idx];
            (*target_reshaped)[new_idx] = target_ids[orig_idx];
        }
    }
}

#endif