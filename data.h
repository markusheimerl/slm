#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t get_corpus_size(const char* filename);
char* load_corpus_chunk(const char* filename, size_t offset, size_t chunk_size, size_t* loaded_size);
void generate_sequences(unsigned char* input_tokens, unsigned char* target_tokens, int num_sequences, int seq_len, char* corpus, size_t corpus_size);

#endif