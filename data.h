#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* text;
    size_t text_size;
} TextData;

// Load text data from file
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
    
    printf("Loaded %zu bytes of text\n", data->text_size);
    return data;
}

// Free text data
void free_text_data(TextData* data) {
    if (data) {
        free(data->text);
        free(data);
    }
}

#endif