#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <ctype.h>
#include <unistd.h>

/*
 * Configuration settings
 */
typedef struct {
    const char* dataset_url;
    const char* temp_file;
    const char* output_file;
    const char* marker;
    int marker_length;
    int progress_bar_width;
    int min_story_length;
} Config;

static const Config config = {
    .dataset_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    .temp_file = "tinystories_raw.txt",
    .output_file = "data.txt",
    .marker = "<|endoftext|>",
    .marker_length = 13,
    .progress_bar_width = 50,
    .min_story_length = 1450  // Minimum story length in characters
};

/*
 * Dynamic text buffer implementation
 */
typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} Buffer;

Buffer* buffer_create(size_t initial_capacity) {
    Buffer* buffer = malloc(sizeof(Buffer));
    if (!buffer) return NULL;
    
    buffer->data = malloc(initial_capacity);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    
    buffer->length = 0;
    buffer->capacity = initial_capacity;
    return buffer;
}

void buffer_append(Buffer* buffer, const char* data, size_t length) {
    if (buffer->length + length + 1 > buffer->capacity) {
        size_t new_capacity = buffer->capacity * 2 + length;
        char* new_data = realloc(buffer->data, new_capacity);
        if (!new_data) return;
        
        buffer->data = new_data;
        buffer->capacity = new_capacity;
    }
    
    memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
}

void buffer_clear(Buffer* buffer) {
    buffer->length = 0;
    if (buffer->capacity > 0) {
        buffer->data[0] = '\0';
    }
}

void buffer_free(Buffer* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}

/*
 * String manipulation utilities
 */
void normalize_whitespace(char* str, size_t length) {
    for (size_t i = 0; i < length; i++) {
        if (str[i] == '\n' || str[i] == '\r') {
            str[i] = ' ';
        }
    }
}

void trim_whitespace(char* str) {
    if (!str) return;
    
    // Trim leading spaces
    char* start = str;
    while (isspace((unsigned char)*start)) start++;
    
    // All spaces?
    if (*start == 0) {
        *str = 0;
        return;
    }
    
    // Trim trailing spaces
    char* end = str + strlen(str) - 1;
    while (end > start && isspace((unsigned char)*end)) end--;
    *(end + 1) = 0;
    
    // Move if needed
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

/*
 * Progress display utilities
 */
void display_progress_bar(const char* label, double percentage) {
    percentage = percentage > 1.0 ? 1.0 : percentage;
    
    int filled_width = (int)(config.progress_bar_width * percentage);
    
    printf("\r%s: [", label);
    for (int i = 0; i < config.progress_bar_width; i++) {
        printf(i < filled_width ? "=" : " ");
    }
    printf("] %.1f%%", percentage * 100);
    fflush(stdout);
}

/*
 * CURL callback functions
 */
size_t write_data_callback(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

int download_progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, 
                              curl_off_t ultotal, curl_off_t ulnow) {
    (void)clientp; (void)ultotal; (void)ulnow;
    
    if (dltotal <= 0) return 0;
    
    double percentage = (double)dlnow / (double)dltotal;
    display_progress_bar("Downloading", percentage);
    
    printf(" (%.2f/%.2f MB)", 
           (double)dlnow / 1048576, 
           (double)dltotal / 1048576);
    
    return 0;
}

/*
 * Dataset handling functions
 */
int download_dataset(void) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        return 0;
    }
    
    FILE* file = fopen(config.temp_file, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", config.temp_file);
        curl_easy_cleanup(curl);
        return 0;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, config.dataset_url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, download_progress_callback);
    
    CURLcode result = curl_easy_perform(curl);
    fclose(file);
    curl_easy_cleanup(curl);
    
    if (result != CURLE_OK) {
        fprintf(stderr, "\nDownload failed: %s\n", curl_easy_strerror(result));
        return 0;
    }
    
    printf("\nDownload complete.\n");
    return 1;
}

void process_story(FILE* output_file, Buffer* story, long* stories_written, long* stories_filtered) {
    if (story->length == 0) return;
    
    normalize_whitespace(story->data, story->length);
    trim_whitespace(story->data);
    
    size_t trimmed_length = strlen(story->data);
    if (trimmed_length >= config.min_story_length) {
        fprintf(output_file, "%s\n", story->data);
        (*stories_written)++;
    } else {
        (*stories_filtered)++;
    }
}

int process_dataset(void) {
    printf("Processing stories...\n");
    
    FILE* input_file = fopen(config.temp_file, "r");
    if (!input_file) {
        fprintf(stderr, "Failed to open downloaded file: %s\n", config.temp_file);
        return 0;
    }
    
    FILE* output_file = fopen(config.output_file, "w");
    if (!output_file) {
        fprintf(stderr, "Failed to open output file: %s\n", config.output_file);
        fclose(input_file);
        return 0;
    }
    
    Buffer* story = buffer_create(4096);
    if (!story) {
        fprintf(stderr, "Failed to allocate memory for story buffer\n");
        fclose(input_file);
        fclose(output_file);
        return 0;
    }
    
    char* line = NULL;
    size_t line_capacity = 0;
    ssize_t line_length;
    
    long story_count = 0;
    long stories_written = 0;
    long stories_filtered = 0;
    
    while ((line_length = getline(&line, &line_capacity, input_file)) != -1) {
        char* marker_pos = strstr(line, config.marker);
        
        if (marker_pos) {
            // Append content before marker to current story
            size_t before_marker = marker_pos - line;
            if (before_marker > 0) {
                buffer_append(story, line, before_marker);
            }
            
            // Process the completed story
            process_story(output_file, story, &stories_written, &stories_filtered);
            story_count++;
            
            // Reset for next story
            buffer_clear(story);
            
            // Process content after marker (start of next story)
            char* next_story_start = marker_pos + config.marker_length;
            size_t remaining = line_length - (next_story_start - line);
            
            if (remaining > 0) {
                buffer_append(story, next_story_start, remaining);
            }
        } else {
            // Append to current story
            buffer_append(story, line, line_length);
        }
    }
    
    // Process final story if present
    if (story->length > 0) {
        process_story(output_file, story, &stories_written, &stories_filtered);
        story_count++;
    }
    
    // Clean up resources
    buffer_free(story);
    free(line);
    fclose(input_file);
    fclose(output_file);
    
    // Report results
    printf("\nProcessing complete:\n");
    printf("- Total stories found: %ld\n", story_count);
    printf("- Stories under %d characters (filtered out): %ld\n", 
           config.min_story_length, stories_filtered);
    printf("- Stories kept: %ld\n", stories_written);
    printf("- Output written to %s\n", config.output_file);
    
    return 1;
}

/*
 * Main program
 */
int main(void) {
    curl_global_init(CURL_GLOBAL_ALL);
    
    printf("Starting dataset download and processing...\n");
    
    int success = download_dataset() && process_dataset();
    
    // Clean up
    if (success) {
        remove(config.temp_file);
    }
    
    curl_global_cleanup();
    return success ? 0 : 1;
}