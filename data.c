#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <ctype.h>
#include <unistd.h>

#define DATASET_URL "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
#define TEMP_FILE "tinystories_raw.txt"
#define OUTPUT_FILE "data.txt"
#define MARKER "<|endoftext|>"
#define MARKER_LEN 13
#define PROGRESS_BAR_WIDTH 50
#define MIN_LENGTH 1450  // Minimum story length in characters

typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} Buffer;

// Initialize an empty buffer
Buffer* buffer_init(size_t initial_capacity) {
    Buffer* buffer = malloc(sizeof(Buffer));
    buffer->data = malloc(initial_capacity);
    buffer->length = 0;
    buffer->capacity = initial_capacity;
    return buffer;
}

// Ensure buffer has enough space for new data
void buffer_ensure_capacity(Buffer* buffer, size_t additional) {
    if (buffer->length + additional > buffer->capacity) {
        size_t new_capacity = buffer->capacity * 2 + additional;
        buffer->data = realloc(buffer->data, new_capacity);
        buffer->capacity = new_capacity;
    }
}

// Append data to buffer
void buffer_append(Buffer* buffer, const char* data, size_t length) {
    buffer_ensure_capacity(buffer, length);
    memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
}

// Free buffer
void buffer_free(Buffer* buffer) {
    free(buffer->data);
    free(buffer);
}

// Write callback for curl
size_t write_callback(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Progress callback for curl
int progress_callback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    (void)clientp; (void)ultotal; (void)ulnow;
    
    if (dltotal <= 0) return 0;
    
    double percentage = (double)dlnow / (double)dltotal;
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    printf("\rDownloading: [");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        printf(i < filled_width ? "=" : " ");
    }
    printf("] %.1f%% (%.2f/%.2f MB)", 
           percentage * 100, 
           (double)dlnow / 1048576, 
           (double)dltotal / 1048576);
    
    fflush(stdout);
    return 0;
}

// Print a progress bar
void print_progress(long current, long total, const char* phase) {
    double percentage = (double)current / (double)total;
    percentage = percentage > 1.0 ? 1.0 : percentage;
    
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    printf("\r%s: [", phase);
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        printf(i < filled_width ? "=" : " ");
    }
    printf("] %.1f%% (%ld/%ld)", percentage * 100, current, total);
    fflush(stdout);
}

// Trim leading/trailing whitespace
void trim(char *str) {
    if (!str) return;
    
    // Trim leading space
    char *start = str;
    while (isspace((unsigned char)*start)) start++;
    
    // All spaces?
    if (*start == 0) {
        *str = 0;
        return;
    }
    
    // Trim trailing space
    char *end = str + strlen(str) - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    *(end + 1) = 0;
    
    // Move if needed
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

// Normalize whitespace in a story
void normalize_whitespace(char *str, size_t length) {
    for (size_t i = 0; i < length; i++) {
        if (str[i] == '\n' || str[i] == '\r') {
            str[i] = ' ';
        }
    }
}

// Download dataset from URL
int download_dataset() {
    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        return 0;
    }
    
    FILE *fp = fopen(TEMP_FILE, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing\n");
        curl_easy_cleanup(curl);
        return 0;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, DATASET_URL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, NULL);
    
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "\nDownload failed: %s\n", curl_easy_strerror(res));
        return 0;
    }
    
    printf("\nDownload complete.\n");
    return 1;
}

// Process dataset and extract stories
int process_dataset() {
    printf("Processing stories...\n");
    
    FILE *input = fopen(TEMP_FILE, "r");
    if (!input) {
        fprintf(stderr, "Failed to open downloaded file\n");
        return 0;
    }
    
    FILE *output = fopen(OUTPUT_FILE, "w");
    if (!output) {
        fprintf(stderr, "Failed to open output file\n");
        fclose(input);
        return 0;
    }
    
    Buffer* story = buffer_init(4096);
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    
    long story_count = 0;
    long stories_written = 0;
    long stories_filtered = 0;
    
    while ((line_len = getline(&line, &line_cap, input)) != -1) {
        char *marker_pos = strstr(line, MARKER);
        
        if (marker_pos) {
            // Process content before marker
            size_t before_marker = marker_pos - line;
            if (before_marker > 0) {
                buffer_append(story, line, before_marker);
            }
            
            // Process current story
            if (story->length > 0) {
                story->data[story->length] = '\0';
                normalize_whitespace(story->data, story->length);
                trim(story->data);
                
                size_t trimmed_length = strlen(story->data);
                if (trimmed_length >= MIN_LENGTH) {
                    fprintf(output, "%s\n", story->data);
                    stories_written++;
                } else {
                    stories_filtered++;
                }
                
                story_count++;
            }
            
            // Reset for next story
            story->length = 0;
            
            // Process content after marker
            char *next_story_start = marker_pos + MARKER_LEN;
            size_t remaining = line_len - (next_story_start - line);
            
            if (remaining > 0) {
                buffer_append(story, next_story_start, remaining);
            }
        } else {
            // Append to current story
            buffer_append(story, line, line_len);
        }
    }
    
    // Process final story if present
    if (story->length > 0) {
        story->data[story->length] = '\0';
        normalize_whitespace(story->data, story->length);
        trim(story->data);
        
        size_t trimmed_length = strlen(story->data);
        if (trimmed_length >= MIN_LENGTH) {
            fprintf(output, "%s\n", story->data);
            stories_written++;
        } else {
            stories_filtered++;
        }
        
        story_count++;
    }
    
    buffer_free(story);
    free(line);
    fclose(input);
    fclose(output);
    
    printf("\nProcessing complete:\n");
    printf("- Total stories found: %ld\n", story_count);
    printf("- Stories under %d characters (filtered out): %ld\n", MIN_LENGTH, stories_filtered);
    printf("- Stories kept: %ld\n", stories_written);
    printf("- Output written to %s\n", OUTPUT_FILE);
    
    return 1;
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);
    
    printf("Downloading dataset...\n");
    if (!download_dataset()) {
        curl_global_cleanup();
        return 1;
    }
    
    if (!process_dataset()) {
        curl_global_cleanup();
        return 1;
    }
    
    remove(TEMP_FILE);
    curl_global_cleanup();
    return 0;
}