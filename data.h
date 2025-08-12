#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function prototypes
char* load_corpus(const char* filename, size_t* corpus_size);
void generate_char_sequences_from_corpus(unsigned char** input_chars, unsigned char** target_chars, int num_sequences, int seq_len, char* corpus, size_t corpus_size);

#endif