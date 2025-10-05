#include "data.h"

// Load the text corpus from a file
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

    if (*corpus_size == 0) {
        printf("Error: Corpus file is empty: %s\n", filename);
        fclose(file);
        return NULL;
    }

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

// Generate all non-overlapping character sequences from corpus in random order
void generate_char_sequences_from_corpus(unsigned char** input_chars, unsigned char** target_chars, int num_sequences, int seq_len, char* corpus, size_t corpus_size) {
    // Calculate the maximum number of non-overlapping sequences we can extract
    int max_sequences = (corpus_size - 1) / seq_len;
    
    if (num_sequences > max_sequences) {
        printf("Error: Requested %d sequences but corpus only has %d non-overlapping sequences of length %d\n", num_sequences, max_sequences, seq_len);
        printf("Corpus size: %zu, needed: %zu\n", corpus_size, (size_t)(num_sequences * seq_len + 1));
        exit(1);
    }
    
    // Create array of sequence starting positions for non-overlapping sequences
    int* sequence_positions = (int*)malloc(max_sequences * sizeof(int));
    for (int i = 0; i < max_sequences; i++) {
        sequence_positions[i] = i * seq_len;
    }
    
    // Fisher-Yates shuffle to create random permutation
    for (int i = max_sequences - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = sequence_positions[i];
        sequence_positions[i] = sequence_positions[j];
        sequence_positions[j] = temp;
    }
    
    // Fill sequences using the shuffled positions
    for (int seq = 0; seq < num_sequences; seq++) {
        size_t start_pos = sequence_positions[seq];
        
        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            (*input_chars)[idx]  = (unsigned char)corpus[start_pos + t];
            (*target_chars)[idx] = (unsigned char)corpus[start_pos + t + 1];
        }
    }
    
    free(sequence_positions);
}