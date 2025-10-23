#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Function prototypes
char* load_corpus(const char* filename, size_t* corpus_size);
void extract_bos_sections(char* corpus, size_t corpus_size, unsigned char** input_tokens, unsigned char** target_tokens, int* num_sections, int seq_len);

#endif