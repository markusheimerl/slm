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

// Generate character sequences from pre-loaded corpus
void generate_char_sequences_from_corpus(unsigned char** input_chars, unsigned char** target_chars, int num_sequences, int seq_len, char* corpus, size_t corpus_size) {
    if (corpus_size < (size_t)(seq_len + 1)) {
        printf("Error: Corpus too small. Need at least %d characters, got %zu\n", seq_len + 1, corpus_size);
        exit(1);
    }

    size_t usable_corpus_size = corpus_size - seq_len;

    for (int seq = 0; seq < num_sequences; seq++) {
        size_t start_pos = rand() % usable_corpus_size;

        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            (*input_chars)[idx]  = (unsigned char)corpus[start_pos + t];
            (*target_chars)[idx] = (unsigned char)corpus[start_pos + t + 1];
        }
    }
}