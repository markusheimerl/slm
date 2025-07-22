#include "data.h"

// Callback function for libcurl to write data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, struct DownloadBuffer* buffer) {
    size_t realsize = size * nmemb;
    
    // Ensure we have enough space
    while (buffer->size + realsize + 1 > buffer->allocated) {
        size_t new_size = buffer->allocated * 2;
        if (new_size < buffer->size + realsize + 1) {
            new_size = buffer->size + realsize + 1;
        }
        
        char* new_data = (char*)realloc(buffer->data, new_size);
        if (!new_data) {
            return 0; // Out of memory
        }
        buffer->data = new_data;
        buffer->allocated = new_size;
    }
    
    memcpy(buffer->data + buffer->size, contents, realsize);
    buffer->size += realsize;
    buffer->data[buffer->size] = '\0';
    
    return realsize;
}

// Extract title from Project Gutenberg text
char* extract_title(const char* content) {
    const char* title_start = strstr(content, "Title:");
    if (!title_start) {
        title_start = strstr(content, "TITLE:");
    }
    if (!title_start) return NULL;
    
    title_start += 6; // Skip "Title:" or "TITLE:"
    while (*title_start == ' ' || *title_start == '\t') title_start++; // Skip whitespace
    
    const char* title_end = strchr(title_start, '\n');
    if (!title_end) title_end = title_start + strlen(title_start);
    
    size_t title_len = title_end - title_start;
    if (title_len > 50) title_len = 50; // Limit title length
    if (title_len == 0) return NULL;
    
    char* title = (char*)malloc(title_len + 1);
    if (!title) return NULL;
    
    strncpy(title, title_start, title_len);
    title[title_len] = '\0';
    
    // Remove carriage returns and clean up
    for (size_t i = 0; i < title_len; i++) {
        if (title[i] == '\r' || title[i] == '\n') {
            title[i] = '\0';
            break;
        }
    }
    
    return title;
}

// Extract author from Project Gutenberg text
char* extract_author(const char* content) {
    const char* author_start = strstr(content, "Author:");
    if (!author_start) {
        author_start = strstr(content, "AUTHOR:");
    }
    if (!author_start) return NULL;
    
    author_start += 7; // Skip "Author:" or "AUTHOR:"
    while (*author_start == ' ' || *author_start == '\t') author_start++; // Skip whitespace
    
    const char* author_end = strchr(author_start, '\n');
    if (!author_end) author_end = author_start + strlen(author_start);
    
    size_t author_len = author_end - author_start;
    if (author_len > 30) author_len = 30; // Limit author length
    if (author_len == 0) return NULL;
    
    char* author = (char*)malloc(author_len + 1);
    if (!author) return NULL;
    
    strncpy(author, author_start, author_len);
    author[author_len] = '\0';
    
    // Remove carriage returns and clean up
    for (size_t i = 0; i < author_len; i++) {
        if (author[i] == '\r' || author[i] == '\n') {
            author[i] = '\0';
            break;
        }
    }
    
    return author;
}

// Download a single book from Project Gutenberg
int download_book(int book_id, int process_id, const char* output_file, 
                  const char* temp_dir, const char* lock_file, 
                  const char* size_file, const char* book_count_file,
                  long target_size_bytes) {
    
    // Construct URL
    char url[256];
    snprintf(url, sizeof(url), "https://www.gutenberg.org/files/%d/%d-0.txt", book_id, book_id);
    
    // Initialize curl
    CURL* curl = curl_easy_init();
    if (!curl) return 0;
    
    // Initialize download buffer
    struct DownloadBuffer buffer;
    buffer.data = (char*)malloc(8192);
    buffer.size = 0;
    buffer.allocated = 8192;
    
    if (!buffer.data) {
        curl_easy_cleanup(curl);
        return 0;
    }
    
    // Set curl options
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 8L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "SLM-Corpus-Downloader/1.0");
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    
    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    // Check for errors
    if (res != CURLE_OK || buffer.size < 1000) {
        free(buffer.data);
        return 0;
    }
    
    // Basic validation - check if it's HTML
    if (strstr(buffer.data, "<html") || strstr(buffer.data, "<!DOCTYPE") || strstr(buffer.data, "<!doctype")) {
        free(buffer.data);
        return 0;
    }
    
    // Extract title and author
    char* title = extract_title(buffer.data);
    char* author = extract_author(buffer.data);
    
    // Create fallback title if needed
    if (!title) {
        title = (char*)malloc(32);
        if (title) {
            snprintf(title, 32, "Book_%d", book_id);
        }
    }
    
    // Create temporary file
    char temp_filename[512];
    snprintf(temp_filename, sizeof(temp_filename), "%s/book_%d_%d.txt", temp_dir, process_id, book_id);
    
    FILE* temp_file = fopen(temp_filename, "w");
    if (!temp_file) {
        free(buffer.data);
        if (title) free(title);
        if (author) free(author);
        return 0;
    }
    
    // Write content to temp file - ensure it ends with newline
    size_t written = fwrite(buffer.data, 1, buffer.size, temp_file);
    
    // Add newline if content doesn't end with one
    if (buffer.size > 0 && buffer.data[buffer.size - 1] != '\n') {
        fwrite("\n", 1, 1, temp_file);
    }
    
    fclose(temp_file);
    
    if (written != buffer.size) {
        unlink(temp_filename);
        free(buffer.data);
        if (title) free(title);
        if (author) free(author);
        return 0;
    }
    
    // Thread-safe append to main file using flock
    int success = 0;
    int lock_fd = open(lock_file, O_CREAT | O_WRONLY, 0666);
    if (lock_fd != -1) {
        if (flock(lock_fd, LOCK_EX) == 0) {
            // Append temp file to main file
            FILE* main_file = fopen(output_file, "a");
            if (main_file) {
                FILE* temp_read = fopen(temp_filename, "r");
                if (temp_read) {
                    char chunk[4096];
                    size_t bytes_read;
                    while ((bytes_read = fread(chunk, 1, sizeof(chunk), temp_read)) > 0) {
                        fwrite(chunk, 1, bytes_read, main_file);
                    }
                    fclose(temp_read);
                    
                    // Ensure separation between books with an additional newline
                    fwrite("\n", 1, 1, main_file);
                    
                    // Update size and count
                    fflush(main_file);
                    fclose(main_file);
                    
                    struct stat st;
                    if (stat(output_file, &st) == 0) {
                        FILE* size_f = fopen(size_file, "w");
                        if (size_f) {
                            fprintf(size_f, "%ld", st.st_size);
                            fclose(size_f);
                        }
                        
                        FILE* count_f = fopen(book_count_file, "r");
                        int current_count = 0;
                        if (count_f) {
                            fscanf(count_f, "%d", &current_count);
                            fclose(count_f);
                        }
                        
                        count_f = fopen(book_count_file, "w");
                        if (count_f) {
                            fprintf(count_f, "%d", current_count + 1);
                            fclose(count_f);
                        }
                        
                        // Report progress
                        long current_mb = st.st_size / (1024 * 1024);
                        long target_mb = target_size_bytes / (1024 * 1024);
                        
                        if (author && title) {
                            printf("P%d Book %d: \"%s\" by %s → %ldMB / %ldMB (%d books)\n", 
                                   process_id, book_id, title, author, current_mb, target_mb, current_count + 1);
                        } else if (title) {
                            printf("P%d Book %d: \"%s\" → %ldMB / %ldMB (%d books)\n", 
                                   process_id, book_id, title, current_mb, target_mb, current_count + 1);
                        }
                        
                        success = 1;
                    }
                } else {
                    fclose(main_file);
                }
            }
            flock(lock_fd, LOCK_UN);
        }
        close(lock_fd);
    }
    
    // Clean up
    unlink(temp_filename);
    free(buffer.data);
    if (title) free(title);
    if (author) free(author);
    
    return success;
}

// Worker process function
void worker_process(int process_id, int start_id, int step, const char* output_file,
                   const char* temp_dir, const char* lock_file, 
                   const char* size_file, const char* book_count_file,
                   long target_size_bytes) {
    
    int book_id = start_id;
    int books_downloaded = 0;
    
    while (book_id <= 50000) {
        // Check if target size reached
        FILE* size_f = fopen(size_file, "r");
        long current_size = 0;
        if (size_f) {
            fscanf(size_f, "%ld", &current_size);
            fclose(size_f);
        }
        
        if (current_size >= target_size_bytes) {
            printf("P%d: Target size reached, stopping\n", process_id);
            break;
        }
        
        if (download_book(book_id, process_id, output_file, temp_dir, lock_file, 
                         size_file, book_count_file, target_size_bytes)) {
            books_downloaded++;
        } else {
            printf("P%d Book %d: skip\n", process_id, book_id);
        }
        
        book_id += step;
        usleep(100000); // Sleep 0.1 seconds
    }
    
    printf("P%d: Downloaded %d books\n", process_id, books_downloaded);
}

// Download Project Gutenberg corpus
int download_corpus(const char* filename, int target_size_mb) {
    printf("Corpus file not found. Downloading Project Gutenberg books...\n");
    
    const int num_processes = 4;
    long target_size_bytes = (long)target_size_mb * 1024 * 1024;
    
    // Initialize libcurl
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        printf("Error: Failed to initialize libcurl\n");
        return 0;
    }
    
    // Create temporary directory
    char temp_dir[] = "/tmp/slm_download_XXXXXX";
    if (!mkdtemp(temp_dir)) {
        printf("Error: Could not create temporary directory\n");
        curl_global_cleanup();
        return 0;
    }
    
    // Create coordination files
    char lock_file[512], size_file[512], book_count_file[512];
    snprintf(lock_file, sizeof(lock_file), "%s/write.lock", temp_dir);
    snprintf(size_file, sizeof(size_file), "%s/current_size", temp_dir);
    snprintf(book_count_file, sizeof(book_count_file), "%s/book_count", temp_dir);
    
    // Initialize files
    FILE* f = fopen(size_file, "w");
    if (f) { fprintf(f, "0"); fclose(f); }
    
    f = fopen(book_count_file, "w");
    if (f) { fprintf(f, "0"); fclose(f); }
    
    f = fopen(filename, "w");
    if (f) fclose(f);
    
    printf("Downloading Project Gutenberg books to reach %dMB using %d processes...\n", 
           target_size_mb, num_processes);
    
    // Fork worker processes
    pid_t pids[num_processes];
    for (int i = 0; i < num_processes; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            // Child process
            worker_process(i + 1, i + 1, num_processes, filename, temp_dir, 
                          lock_file, size_file, book_count_file, target_size_bytes);
            exit(0);
        } else if (pids[i] < 0) {
            printf("Error: Failed to fork process %d\n", i + 1);
        }
    }
    
    // Wait for all processes to complete
    for (int i = 0; i < num_processes; i++) {
        if (pids[i] > 0) {
            waitpid(pids[i], NULL, 0);
        }
    }
    
    // Final summary
    struct stat st;
    int final_books = 0;
    
    if (stat(filename, &st) == 0) {
        f = fopen(book_count_file, "r");
        if (f) {
            fscanf(f, "%d", &final_books);
            fclose(f);
        }
        
        printf("\nComplete: %d books, %ld bytes\n", final_books, st.st_size);
    }
    
    // Cleanup
    char cleanup_cmd[1024];
    snprintf(cleanup_cmd, sizeof(cleanup_cmd), "rm -rf %s", temp_dir);
    system(cleanup_cmd);
    
    curl_global_cleanup();
    
    return (final_books > 0) ? 1 : 0;
}

// Load text corpus from file, download if not available
char* load_corpus(const char* filename, size_t* corpus_size, int target_size_bytes) {
    FILE* file = fopen(filename, "r");
    
    // If file doesn't exist, try to download it
    if (!file) {
        printf("Corpus file not found: %s\n", filename);
        
        // Only attempt download for the default gutenberg corpus
        if (strcmp(filename, "gutenberg_corpus.txt") == 0) {
            if (download_corpus(filename, target_size_bytes / (1024 * 1024)) == 0) {
                // Try to open the file again after download
                file = fopen(filename, "r");
            }
        }
        
        if (!file) {
            printf("Error: Could not open corpus file: %s\n", filename);
            return NULL;
        }
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *corpus_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Check if file is empty
    if (*corpus_size == 0) {
        printf("Error: Corpus file is empty: %s\n", filename);
        fclose(file);
        return NULL;
    }
    
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

// Generate character sequences from pre-loaded corpus
void generate_char_sequences_from_corpus(unsigned char** input_chars, unsigned char** target_chars, 
                                        int num_sequences, int seq_len, char* corpus, size_t corpus_size) {
    // Ensure we have enough data
    if (corpus_size < (size_t)(seq_len + 1)) {
        printf("Error: Corpus too small. Need at least %d characters, got %zu\n", 
               seq_len + 1, corpus_size);
        exit(1);
    }
    
    size_t usable_corpus_size = corpus_size - seq_len;
    
    // Generate sequences from random locations
    for (int seq = 0; seq < num_sequences; seq++) {
        size_t start_pos = rand() % usable_corpus_size;
        
        for (int t = 0; t < seq_len; t++) {
            int idx = seq * seq_len + t;
            (*input_chars)[idx] = (unsigned char)corpus[start_pos + t];
            (*target_chars)[idx] = (unsigned char)corpus[start_pos + t + 1];
        }
    }
}