#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>

#define DATASET_URL "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
#define TEMP_FILE "tinystories_raw.txt"
#define OUTPUT_FILE "data.txt"
#define MARKER "<|endoftext|>"
#define MARKER_LEN 13
#define PROGRESS_BAR_WIDTH 50

// Structure to track download progress
typedef struct {
    int initialized;
    int last_percent;
} ProgressData;

// Write callback for curl
size_t write_callback(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Progress callback for curl
int progress_callback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    (void)ultotal;  // Mark as unused
    (void)ulnow;    // Mark as unused
    
    ProgressData *progress = (ProgressData*)clientp;
    
    // Skip if we don't have a content length yet
    if (dltotal <= 0) {
        if (!progress->initialized) {
            printf("\rDownloading: [%*s] 0.0%% (0.00 MB)", PROGRESS_BAR_WIDTH, "");
            fflush(stdout);
            progress->initialized = 1;
        }
        return 0;
    }
    
    // Calculate percentage
    double percentage = (double)dlnow / (double)dltotal;
    int percent = (int)(percentage * 100);
    
    // Only update if percent changed or first time
    if (!progress->initialized || percent != progress->last_percent) {
        progress->initialized = 1;
        progress->last_percent = percent;
        
        int filled_width = (int)(PROGRESS_BAR_WIDTH * percentage);
        
        // Print progress bar
        printf("\rDownloading: [");
        for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {
            if (i < filled_width) printf("=");
            else printf(" ");
        }
        printf("] %.1f%% (%.2f/%.2f MB)", 
               percentage * 100, 
               (double)dlnow / 1048576,  // Convert to MB
               (double)dltotal / 1048576);
        fflush(stdout);
    }
    
    return 0; // Return 0 to continue transfer
}

// Function to print a processing progress bar
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

// Function to count markers in a file
long count_stories(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;
    
    long count = 0;
    long total_bytes = 0;
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    
    struct stat st;
    stat(filename, &st);
    long file_size = st.st_size;
    
    printf("Counting stories...\n");
    printf("\rCounting: [%*s] 0.0%% (0 markers)", PROGRESS_BAR_WIDTH, "");
    fflush(stdout);
    
    // Read line by line
    while ((line_len = getline(&line, &line_cap, file)) != -1) {
        total_bytes += line_len;
        
        // Check for markers in this line
        char *pos = line;
        while ((pos = strstr(pos, MARKER)) != NULL) {
            count++;
            pos += MARKER_LEN;
        }
        
        // Update progress based on file position
        if (total_bytes % (10 * 1024 * 1024) == 0 || total_bytes >= file_size - 1) {
            double percentage = (double)total_bytes / (double)file_size;
            if (percentage > 1.0) percentage = 1.0;
            
            print_progress_bar(count, count > 0 ? count : 1, "Counting");
        }
    }
    
    free(line);
    fclose(file);
    printf("\nFound %ld stories.\n", count);
    return count;
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
    
    // Initialize progress tracking
    ProgressData progress = {0, 0};
    
    curl_easy_setopt(curl, CURLOPT_URL, DATASET_URL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);
    
    res = curl_easy_perform(curl);
    fclose(fp);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "\nDownload failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return 1;
    }
    
    printf("\nDownload complete.\n");
    curl_easy_cleanup(curl);
    
    // Step 2: Count total number of stories
    long total_stories = count_stories(TEMP_FILE);
    
    if (total_stories == 0) {
        fprintf(stderr, "No stories found in the file or file could not be read.\n");
        return 1;
    }
    
    // Step 3: Process the file by reading line by line
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
    
    // Initialize with empty progress bar (0%)
    print_progress_bar(0, total_stories, "Processing");
    
    // Buffer for story content
    char *story = NULL;
    size_t story_capacity = 0;
    size_t story_length = 0;
    
    // Buffer for reading lines
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len;
    
    long story_count = 0;
    int found_marker = 0;
    
    // Process line by line
    while ((line_len = getline(&line, &line_cap, input)) != -1) {
        // Update progress every 100 stories
        if (story_count % 100 == 0) {
            print_progress_bar(story_count, total_stories, "Processing");
        }
        
        // Check if the line contains our marker
        char *marker_pos = strstr(line, MARKER);
        
        if (marker_pos) {
            // Line contains the marker - process everything before it
            found_marker = 1;
            
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
            
            // Write to output if not empty
            if (story_length > 0) {
                fprintf(output, "%s\n", story);
                story_count++;
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
        
        // Write to output if not empty
        if (story_length > 0) {
            fprintf(output, "%s\n", story);
            story_count++;
        }
    }
    
    // Print final progress
    print_progress_bar(story_count, total_stories, "Processing");
    printf("\n");
    
    // Clean up
    free(line);
    free(story);
    fclose(input);
    fclose(output);
    
    printf("\nCompleted! Processed %ld stories.\n", story_count);
    printf("Output written to %s\n", OUTPUT_FILE);
    
    if (!found_marker) {
        printf("Warning: No end-of-text markers found in the file. Check if the format is correct.\n");
    }
    
    // Clean up
    remove(TEMP_FILE);
    curl_global_cleanup();
    
    return 0;
}