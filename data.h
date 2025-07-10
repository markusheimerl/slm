#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CHAR_VALUE 256  // Extended ASCII characters 0-255
#define EMBED_DIM 512       // Embedding dimension

// Fixed random embedding matrix for characters
float embedding_matrix[MAX_CHAR_VALUE][EMBED_DIM];
int embedding_initialized = 0;

// Initialize fixed random embeddings for all characters
void init_embeddings() {
    if (embedding_initialized) return;
    
    for (int i = 0; i < MAX_CHAR_VALUE; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            // Xavier/Glorot initialization for embeddings
            float scale = sqrt(2.0f / EMBED_DIM);
            embedding_matrix[i][j] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    embedding_initialized = 1;
}

// Get embedding for a character
void get_char_embedding(unsigned char c, float* embedding) {
    if (!embedding_initialized) {
        init_embeddings();
    }
    
    int char_idx = (int)c;
    
    for (int i = 0; i < EMBED_DIM; i++) {
        embedding[i] = embedding_matrix[char_idx][i];
    }
}

// Create one-hot encoding for a character
void create_one_hot(unsigned char c, float* one_hot) {
    // Initialize all to 0
    for (int i = 0; i < MAX_CHAR_VALUE; i++) {
        one_hot[i] = 0.0f;
    }
    
    // Set the appropriate index to 1
    int char_idx = (int)c;
    one_hot[char_idx] = 1.0f;
}

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

// Generate text sequence data from corpus
void generate_text_sequence_data(float** X, float** y, int num_sequences, int seq_len, 
                                int input_dim, int output_dim, const char* corpus_filename) {
    // Initialize embeddings
    init_embeddings();
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus(corpus_filename, &corpus_size);
    if (!corpus) {
        exit(1);
    }
    
    // Ensure we have enough data
    if (corpus_size < (size_t)(num_sequences + seq_len)) {
        printf("Error: Corpus too small. Need at least %d characters, got %zu\n", 
               num_sequences + seq_len, corpus_size);
        free(corpus);
        exit(1);
    }
    
    // Allocate memory for sequences
    *X = (float*)malloc(num_sequences * seq_len * input_dim * sizeof(float));
    *y = (float*)malloc(num_sequences * seq_len * output_dim * sizeof(float));
    
    if (!*X || !*y) {
        printf("Error: Could not allocate memory for sequences\n");
        free(corpus);
        exit(1);
    }
    
    // Generate sequences
    for (int seq = 0; seq < num_sequences; seq++) {
        // Start position for this sequence (staggered by 1 character)
        size_t start_pos = seq;
        
        for (int t = 0; t < seq_len; t++) {
            size_t char_pos = start_pos + t;
            size_t next_char_pos = char_pos + 1;
            
            // Ensure we don't go beyond corpus
            if (next_char_pos >= corpus_size) {
                // Wrap around to beginning
                char_pos = char_pos % corpus_size;
                next_char_pos = (char_pos + 1) % corpus_size;
            }
            
            // Get current character and its embedding
            unsigned char current_char = (unsigned char)corpus[char_pos];
            unsigned char next_char = (unsigned char)corpus[next_char_pos];
            
            // Calculate indices in the data arrays
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            // Get embedding for current character (input)
            get_char_embedding(current_char, &(*X)[x_idx]);
            
            // Create one-hot for next character (target)
            create_one_hot(next_char, &(*y)[y_idx]);
        }
    }
    
    printf("Generated %d sequences of length %d\n", num_sequences, seq_len);
    printf("Input dimensions: %d, Output dimensions: %d\n", input_dim, output_dim);
    
    free(corpus);
}

// Save sequence data to CSV for inspection
void save_text_sequence_data_to_csv(float* X, float* y, int num_sequences, int seq_len, 
                                   int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    fprintf(file, "seq_id,time_step,");
    
    // Input features (embeddings)
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    
    // Output features (one-hot) - only show first 20 for readability
    int show_outputs = (output_dim > 20) ? 20 : output_dim;
    for (int i = 0; i < show_outputs - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d,target_char\n", show_outputs - 1);
    
    // Write data (only first 10 sequences for readability)
    int show_sequences = (num_sequences > 10) ? 10 : num_sequences;
    for (int seq = 0; seq < show_sequences; seq++) {
        for (int t = 0; t < seq_len; t++) {
            int x_idx = seq * seq_len * input_dim + t * input_dim;
            int y_idx = seq * seq_len * output_dim + t * output_dim;
            
            fprintf(file, "%d,%d,", seq, t);
            
            // Input features (embeddings) - only show first 10 for readability
            int show_inputs = (input_dim > 10) ? 10 : input_dim;
            for (int j = 0; j < show_inputs; j++) {
                fprintf(file, "%.6f,", X[x_idx + j]);
            }
            
            // Find target character (index with 1.0 in one-hot)
            int target_char = 0;
            for (int j = 0; j < output_dim; j++) {
                if (y[y_idx + j] == 1.0f) {
                    target_char = j;
                    break;
                }
            }
            
            // Output values (one-hot) - only show first few
            for (int j = 0; j < show_outputs - 1; j++) {
                fprintf(file, "%.0f,", y[y_idx + j]);
            }
            fprintf(file, "%.0f,%c\n", y[y_idx + show_outputs - 1], 
                    (target_char >= 32 && target_char < 127) ? target_char : '?');
        }
    }
    
    fclose(file);
    printf("Text sequence data saved to %s\n", filename);
}

#endif