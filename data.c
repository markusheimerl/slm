#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <curl/curl.h>

// Structure to store memory for curl callbacks
typedef struct {
    char *data;
    size_t size;
} MemoryBuffer;

// Callback function for curl to write received data
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryBuffer *mem = (MemoryBuffer *)userp;
    
    char *ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    
    return realsize;
}

// Read file content into a buffer
char* read_file(const char* path) {
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", path);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate memory for the file content
    char *buffer = malloc(fileSize + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read the file content
    size_t bytesRead = fread(buffer, 1, fileSize, file);
    buffer[bytesRead] = '\0';
    fclose(file);
    
    return buffer;
}

// Download text from a URL
char* download_gutenberg(const char* url, const char* save_path) {
    struct stat file_info;
    
    // Check if file already exists
    if (stat(save_path, &file_info) == 0) {
        printf("File %s already exists, reading from disk...\n", save_path);
        return read_file(save_path);
    }

    // File doesn't exist, download it
    CURL *curl_handle;
    CURLcode res;
    MemoryBuffer chunk = {0};
    
    chunk.data = malloc(1);
    chunk.size = 0;
    
    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle = curl_easy_init();
    
    printf("Downloading %s...\n", url);
    
    if (!curl_handle) {
        fprintf(stderr, "Error: Failed to initialize curl\n");
        free(chunk.data);
        curl_global_cleanup();
        return NULL;
    }
    
    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    
    res = curl_easy_perform(curl_handle);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "Error: curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        free(chunk.data);
        curl_easy_cleanup(curl_handle);
        curl_global_cleanup();
        return NULL;
    }
    
    // Save the downloaded text to a file
    FILE *file = fopen(save_path, "w");
    if (file) {
        fwrite(chunk.data, 1, chunk.size, file);
        fclose(file);
    } else {
        fprintf(stderr, "Error: Could not open file %s for writing\n", save_path);
    }
    
    curl_easy_cleanup(curl_handle);
    curl_global_cleanup();
    
    return chunk.data;
}

// Get combined corpus from multiple Gutenberg texts
char* get_combined_corpus() {
    // Create directory for Gutenberg texts if it doesn't exist
    mkdir("gutenberg_texts", 0755);
    
    // Define the books to download
    const char* books[][2] = {
        {"https://www.gutenberg.org/files/1342/1342-0.txt", "gutenberg_texts/pride_and_prejudice.txt"},
        {"https://www.gutenberg.org/files/84/84-0.txt", "gutenberg_texts/frankenstein.txt"},
        {"https://www.gutenberg.org/files/1661/1661-0.txt", "gutenberg_texts/sherlock_holmes.txt"},
        {"https://www.gutenberg.org/files/2701/2701-0.txt", "gutenberg_texts/moby_dick.txt"},
        {"https://www.gutenberg.org/files/98/98-0.txt", "gutenberg_texts/tale_of_two_cities.txt"},
        {"https://www.gutenberg.org/files/1400/1400-0.txt", "gutenberg_texts/great_expectations.txt"},
        {"https://www.gutenberg.org/files/345/345-0.txt", "gutenberg_texts/dracula.txt"},
        {"https://www.gutenberg.org/files/174/174-0.txt", "gutenberg_texts/dorian_gray.txt"},
        {"https://www.gutenberg.org/files/16/16-0.txt", "gutenberg_texts/peter_pan.txt"},
        {"https://www.gutenberg.org/files/768/768-0.txt", "gutenberg_texts/wuthering_heights.txt"},
        {"https://www.gutenberg.org/files/45/45-0.txt", "gutenberg_texts/anne_of_green_gables.txt"},
        {"https://www.gutenberg.org/files/1260/1260-0.txt", "gutenberg_texts/jane_eyre.txt"}
    };
    
    int num_books = sizeof(books) / sizeof(books[0]);
    char** texts = malloc(num_books * sizeof(char*));
    size_t total_size = 0;
    
    // Download each book and calculate total size
    for (int i = 0; i < num_books; i++) {
        printf("Processing %s...\n", books[i][1]);
        texts[i] = download_gutenberg(books[i][0], books[i][1]);
        if (texts[i]) {
            total_size += strlen(texts[i]) + 2; // +2 for the newlines
        }
    }
    
    // Allocate memory for combined corpus
    char* combined_text = malloc(total_size + 1);
    if (!combined_text) {
        fprintf(stderr, "Error: Memory allocation failed for combined corpus\n");
        for (int i = 0; i < num_books; i++) {
            free(texts[i]);
        }
        free(texts);
        return NULL;
    }
    
    combined_text[0] = '\0';
    
    // Combine all texts
    for (int i = 0; i < num_books; i++) {
        if (texts[i]) {
            strcat(combined_text, texts[i]);
            strcat(combined_text, "\n\n");
            free(texts[i]);
        }
    }
    
    free(texts);
    
    // Save combined corpus
    FILE* file = fopen("gutenberg_texts/combined_corpus.txt", "w");
    if (file) {
        fprintf(file, "%s", combined_text);
        fclose(file);
        printf("Combined corpus saved to gutenberg_texts/combined_corpus.txt\n");
    } else {
        fprintf(stderr, "Error: Could not save combined corpus\n");
    }
    
    printf("Combined corpus size: %zu characters\n", strlen(combined_text));
    
    return combined_text;
}

// Function to load data from file or download if it doesn't exist
char* load_data(const char* filepath) {
    struct stat file_info;
    
    // Check if combined corpus already exists
    if (stat(filepath, &file_info) == 0) {
        printf("Combined corpus already exists, loading from %s\n", filepath);
        return read_file(filepath);
    } else {
        // File doesn't exist, download the corpus
        return get_combined_corpus();
    }
}

int main() {
    char* corpus = load_data("gutenberg_texts/combined_corpus.txt");
    if (corpus) {
        printf("Successfully loaded/downloaded corpus of %zu bytes\n", strlen(corpus));
        free(corpus);
        return 0;
    } else {
        fprintf(stderr, "Failed to load/download corpus\n");
        return 1;
    }
}