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
#define MIN_LENGTH 500  // Minimum story length in characters

// Write callback for curl
size_t write_callback(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Progress callback for curl
int progress_callback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    (void)clientp;  // Mark as unused
    (void)ultotal;  // Mark as unused
    (void)ulnow;    // Mark as unused
    
    if (dltotal <= 0) return 0; // Avoid division by zero
    
    // Calculate percentage
    double percentage = (double)dlnow / (double)dltotal;
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    // Print progress bar
    printf("\rDownloading: [");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        if (i < filled_width) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% (%.2f/%.2f MB)", 
           percentage * 100, 
           (double)dlnow / 1048576, // Convert to MB
           (double)dltotal / 1048576);
    
    fflush(stdout);
    return 0; // Return 0 to continue transfer
}

// Function to print a progress bar
void print_progress_bar(long current, long total, const char* phase) {
    double percentage = (double)current / (double)total;
    if (percentage > 1.0) percentage = 1.0; // Cap at 100%
    
    int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
    
    printf("\r%s: [", phase);
    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
        if (i < filled_width) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% (%ld/%ld)", percentage * 100, current, total);
    fflush(stdout);
}

// Function to trim leading/trailing whitespace
void trim(char *str) {
    if (!str) return;
    
    // Trim leading space
    char *start = str;
    while(isspace((unsigned char)*start)) start++;
    
    // All spaces?
    if(*start == 0) {
        *str = 0;
        return;
    }
    
    // Trim trailing space
    char *end = str + strlen(str) - 1;
    while(end > start && isspace((unsigned char)*end)) end--;
    
    // Write new null terminator
    *(end + 1) = 0;
    
    // Move if needed
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

int main() {
    CURL *curl;
    CURLcode res;
    FILE *fp;
    
    // Initialize curl
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        return 1;
    }
    
    // Step 1: Download the file
    printf("Downloading dataset...\n");
    
    fp = fopen(TEMP_FILE, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing\n");
        curl_easy_cleanup(curl);
        return 1;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, DATASET_URL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, NULL);
    
    res = curl_easy_perform(curl);
    fclose(fp);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "\nDownload failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return 1;
    }
    
    printf("\nDownload complete.\n");
    curl_easy_cleanup(curl);
    
    // Step 2: Process the file by reading line by line
    printf("Processing stories...\n");
    
    FILE *input = fopen(TEMP_FILE, "r");
    if (!input) {
        fprintf(stderr, "Failed to open downloaded file\n");
        return 1;
    }
    
    FILE *output = fopen(OUTPUT_FILE, "w");
    if (!output) {
        fprintf(stderr, "Failed to open output file\n");
        fclose(input);
        return 1;
    }
    
    // Buffer for story content
    char *story = NULL;
    size_t story_capacity = 0;
    size_t story_length = 0;
    
    // Buffer for reading lines
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    
    long story_count = 0;
    long stories_written = 0;
    long stories_filtered = 0;
    
    // Process line by line
    while ((line_len = getline(&line, &line_cap, input)) != -1) {
        // Check if the line contains our marker
        char *marker_pos = strstr(line, MARKER);
        
        if (marker_pos) {
            // Calculate how much content is before the marker
            size_t before_marker = marker_pos - line;
            
            // Ensure story buffer is large enough
            if (story_length + before_marker + 1 > story_capacity) {
                size_t new_capacity = story_length + before_marker + 1024;
                char *new_story = realloc(story, new_capacity);
                if (!new_story) {
                    fprintf(stderr, "Memory allocation error\n");
                    break;
                }
                story = new_story;
                story_capacity = new_capacity;
            }
            
            // Add content before marker to the story
            if (before_marker > 0) {
                memcpy(story + story_length, line, before_marker);
                story_length += before_marker;
            }
            
            // Terminate the story
            story[story_length] = '\0';
            
            // Process the story
            for (size_t i = 0; i < story_length; i++) {
                if (story[i] == '\n' || story[i] == '\r') {
                    story[i] = ' ';
                }
            }
            
            trim(story);
            
            // Only write if the story is long enough
            if (story_length >= MIN_LENGTH) {
                fprintf(output, "%s\n", story);
                stories_written++;
            } else {
                stories_filtered++;
            }
            
            story_count++;
            if (story_count % 10000 == 0) {
                print_progress_bar(story_count, 3000000, "Processing");  // Estimate ~3M stories
            }
            
            // Reset for the next story
            story_length = 0;
            
            // Process any content after the marker (may be the beginning of next story)
            char *next_story_start = marker_pos + MARKER_LEN;
            size_t remaining = line_len - (next_story_start - line);
            
            if (remaining > 0) {
                // Ensure story buffer is large enough
                if (remaining + 1 > story_capacity) {
                    size_t new_capacity = remaining + 1024;
                    char *new_story = realloc(story, new_capacity);
                    if (!new_story) {
                        fprintf(stderr, "Memory allocation error\n");
                        break;
                    }
                    story = new_story;
                    story_capacity = new_capacity;
                }
                
                memcpy(story, next_story_start, remaining);
                story_length = remaining;
            }
        } else {
            // Regular line, add to current story
            
            // Ensure story buffer is large enough
            if (story_length + line_len + 1 > story_capacity) {
                size_t new_capacity = story_length + line_len + 1024;
                char *new_story = realloc(story, new_capacity);
                if (!new_story) {
                    fprintf(stderr, "Memory allocation error\n");
                    break;
                }
                story = new_story;
                story_capacity = new_capacity;
            }
            
            // Add line to the story
            memcpy(story + story_length, line, line_len);
            story_length += line_len;
        }
    }
    
    // Process the last story if we have one
    if (story_length > 0) {
        story[story_length] = '\0';
        
        // Process the story
        for (size_t i = 0; i < story_length; i++) {
            if (story[i] == '\n' || story[i] == '\r') {
                story[i] = ' ';
            }
        }
        
        trim(story);
        
        // Only write if the story is long enough
        if (story_length >= MIN_LENGTH) {
            fprintf(output, "%s\n", story);
            stories_written++;
        } else {
            stories_filtered++;
        }
        
        story_count++;
    }
    
    // Print final progress
    print_progress_bar(story_count, story_count, "Processing");
    printf("\n");
    
    // Clean up
    free(line);
    free(story);
    fclose(input);
    fclose(output);
    remove(TEMP_FILE);
    curl_global_cleanup();
    
    printf("\nProcessing complete:\n");
    printf("- Total stories found: %ld\n", story_count);
    printf("- Stories under %d characters (filtered out): %ld\n", MIN_LENGTH, stories_filtered);
    printf("- Stories kept: %ld\n", stories_written);
    printf("- Output written to %s\n", OUTPUT_FILE);
    
    return 0;
}