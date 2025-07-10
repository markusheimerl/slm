#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load text corpus from file
char* load_corpus(const char* filename, size_t* corpus_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open corpus file: %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *corpus_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate memory and read file
    char* corpus = (char*)malloc((*corpus_size + 1) * sizeof(char));
    if (!corpus) {
        printf("Error: Could not allocate memory for corpus\n");
        fclose(file);
        return NULL;
    }
    
    size_t read_size = fread(corpus, 1, *corpus_size, file);
    corpus[read_size] = '\0';
    *corpus_size = read_size;
    
    fclose(file);
    printf("Loaded corpus: %zu characters\n", *corpus_size);
    return corpus;
}

// Generate character sequence data
void generate_char_sequences(unsigned char** input_chars, unsigned char** target_chars, 
                            int num_sequences, int seq_len, const char* corpus_filename) {
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus(corpus_filename, &corpus_size);
    if (!corpus) {
        exit(1);
    }
    
    // Ensure we have enough data
    if (corpus_size < (size_t)(seq_len + 1)) {
        printf("Error: Corpus too small. Need at least %d characters, got %zu\n", 
               seq_len + 1, corpus_size);
        free(corpus);
        exit(1);
    }
    
    size_t usable_corpus_size = corpus_size - seq_len;
    
    // Debug print
    printf("\n=== SAMPLE SEQUENCES ===\n");
    for (int seq = 0; seq < 3 && seq < num_sequences; seq++) {
        size_t start_pos = rand() % usable_corpus_size;
        printf("Seq %d: \"", seq);
        for (int i = 0; i < 50 && i < seq_len; i++) {
            char c = corpus[start_pos + i];
            if (c >= 32 && c <= 126) printf("%c", c);
            else if (c == '\n') printf("\\n");
            else if (c == '\t') printf("\\t");
            else printf(".");
        }
        printf("...\"\n");
    }
    printf("========================\n\n");
    
    // Allocate memory for sequences (just character indices)
    *input_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    *target_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    
    // Generate sequences
    for (int seq = 0; seq < num_sequences; seq++) {
        size_t start_pos = rand() % usable_corpus_size;
        
        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            (*input_chars)[idx] = (unsigned char)corpus[start_pos + t];
            (*target_chars)[idx] = (unsigned char)corpus[start_pos + t + 1];
        }
    }

    free(corpus);
}

// Save sequences to CSV
void save_sequences_to_csv(unsigned char* input_chars, unsigned char* target_chars, 
                          int num_sequences, int seq_len, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fprintf(file, "seq_id,timestep,input_char,target_char\n");
    
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            fprintf(file, "%d,%d,%d,%d\n", seq, t, 
                   (int)input_chars[idx], (int)target_chars[idx]);
        }
    }
    
    fclose(file);
    printf("Sequences saved to %s\n", filename);
}

#endif