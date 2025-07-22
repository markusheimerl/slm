#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <curl/curl.h>
#include <sys/file.h>
#include <errno.h>

// Structure to hold downloaded content
struct DownloadBuffer {
    char* data;
    size_t size;
    size_t allocated;
};

// Function prototypes
char* extract_title(const char* content);
char* extract_author(const char* content);
int download_book(int book_id, int process_id, const char* output_file, 
                  const char* temp_dir, const char* lock_file, 
                  const char* size_file, const char* book_count_file,
                  long target_size_bytes);
void worker_process(int process_id, int start_id, int step, const char* output_file,
                   const char* temp_dir, const char* lock_file, 
                   const char* size_file, const char* book_count_file,
                   long target_size_bytes);
int download_corpus(const char* filename, int target_size_mb);
char* load_corpus(const char* filename, size_t* corpus_size, int target_size_bytes);
void generate_char_sequences_from_corpus(unsigned char** input_chars, unsigned char** target_chars, 
                                        int num_sequences, int seq_len, char* corpus, size_t corpus_size);

#endif