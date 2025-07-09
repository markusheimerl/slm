#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

typedef struct {
    char* text;
    size_t text_size;
    size_t* book_starts;
    int num_books;
} TextData;

// Find book boundaries by searching for "START OF THE PROJECT GUTENBERG EBOOK"
void find_book_boundaries(TextData* data) {
    const char* marker = "START OF THE PROJECT GUTENBERG EBOOK";
    size_t marker_len = strlen(marker);
    
    // Count occurrences first
    data->num_books = 0;
    for (size_t i = 0; i < data->text_size - marker_len; i++) {
        if (strncmp(&data->text[i], marker, marker_len) == 0) {
            data->num_books++;
        }
    }
    
    if (data->num_books == 0) {
        // Fallback: create artificial boundaries every 100KB
        data->num_books = data->text_size / 100000;
        data->book_starts = (size_t*)malloc(data->num_books * sizeof(size_t));
        for (int i = 0; i < data->num_books; i++) {
            data->book_starts[i] = i * 100000;
        }
        printf("No book markers found, created %d artificial boundaries\n", data->num_books);
        return;
    }
    
    // Allocate and fill book starts
    data->book_starts = (size_t*)malloc(data->num_books * sizeof(size_t));
    int book_idx = 0;
    for (size_t i = 0; i < data->text_size - marker_len; i++) {
        if (strncmp(&data->text[i], marker, marker_len) == 0) {
            data->book_starts[book_idx++] = i;
        }
    }
    
    printf("Found %d book boundaries\n", data->num_books);
}

// Load text data
TextData* load_text_data(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate and read text
    TextData* data = (TextData*)malloc(sizeof(TextData));
    data->text = (char*)malloc(file_size + 1);
    data->text_size = fread(data->text, 1, file_size, file);
    data->text[data->text_size] = '\0';
    
    fclose(file);
    
    // Find book boundaries
    find_book_boundaries(data);
    
    printf("Loaded %zu bytes of text with %d books\n", data->text_size, data->num_books);
    return data;
}

// Get batch of character sequences
void get_batch(TextData* data, int* h_chars, int* h_next_chars,
               int batch_size, int context_length) {
    for (int b = 0; b < batch_size; b++) {
        // Pick a random book start
        int book_idx = rand() % data->num_books;
        size_t start_pos = data->book_starts[book_idx];
        
        // Make sure we don't go past end of text
        if (start_pos + context_length + 1 >= data->text_size) {
            start_pos = data->text_size - context_length - 2;
        }
        
        // Copy characters
        for (int t = 0; t < context_length; t++) {
            h_chars[t * batch_size + b] = (int)(unsigned char)data->text[start_pos + t];
            h_next_chars[t * batch_size + b] = (int)(unsigned char)data->text[start_pos + t + 1];
        }
    }
}

// Free text data
void free_text_data(TextData* data) {
    if (data) {
        free(data->text);
        free(data->book_starts);
        free(data);
    }
}

#endif