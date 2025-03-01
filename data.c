#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <pthread.h>

#define NUM_THREADS 16
#define MAX_BUFFER_SIZE 1024*1024 // 1MB per thread

// Structure to hold thread-specific data
typedef struct {
    int thread_id;
    char *buffer;
    size_t buffer_size;
    size_t buffer_pos;
    pthread_mutex_t *console_mutex;
} ThreadData;

// Mutex for console output
pthread_mutex_t console_mutex = PTHREAD_MUTEX_INITIALIZER;

// The common prompt for all threads
const char *prompt = 
    "Create a training dataset of multi-turn conversations about stories. "
    "Format each example as a single line with <USER> and <ASSISTANT> tags, "
    "using \\\\n to represent newlines within stories. Separate different "
    "training examples with ONLY A SINGLE newline. Make topics diverse and "
    "CREATIVE! Also make the user ask questions about the story within a "
    "training example. Create at least 10 different multi-turn examples.";

// CURL write callback function
size_t writefunc(void *ptr, size_t size, size_t nmemb, void *userdata) {
    ThreadData *thread_data = (ThreadData *)userdata;
    const char *data = (char *)ptr;
    size_t realsize = size * nmemb;
    
    // Search for text_delta in the chunk
    const char *delta_marker = "\"text_delta\",\"text\":\"";
    const char *delta_start = strstr(data, delta_marker);
    
    if (delta_start) {
        delta_start += strlen(delta_marker);
        const char *delta_end = strchr(delta_start, '"');
        
        if (delta_end) {
            // Extract and process the content
            int len = delta_end - delta_start;
            char *content = malloc(len + 1);
            
            if (content) {
                strncpy(content, delta_start, len);
                content[len] = '\0';
                
                // Unescape JSON string
                char *unescaped = malloc(len + 1);
                int j = 0;
                
                for (int i = 0; i < len; i++) {
                    if (content[i] == '\\' && i + 1 < len) {
                        i++;
                        switch (content[i]) {
                            case 'n': unescaped[j++] = '\n'; break;
                            case '\\': unescaped[j++] = '\\'; break;
                            case '"': unescaped[j++] = '"'; break;
                            default: unescaped[j++] = content[i];
                        }
                    } else {
                        unescaped[j++] = content[i];
                    }
                }
                
                unescaped[j] = '\0';
                
                // Append to thread buffer
                size_t unescaped_len = strlen(unescaped);
                if (thread_data->buffer_pos + unescaped_len < thread_data->buffer_size) {
                    memcpy(thread_data->buffer + thread_data->buffer_pos, unescaped, unescaped_len);
                    thread_data->buffer_pos += unescaped_len;
                    thread_data->buffer[thread_data->buffer_pos] = '\0';
                }
                
                free(unescaped);
                free(content);
            }
        }
    }
    
    return realsize;
}

// Process the buffer: normalize newlines and ensure correct formatting
char* process_buffer(const char *buffer) {
    if (!buffer || buffer[0] == '\0') return NULL;
    
    size_t len = strlen(buffer);
    char *output = malloc(len + 8); // Extra space for potential <USER> prefix
    if (!output) return NULL;
    
    // Copy and normalize in one pass
    const char *src = buffer;
    char *dst = output;
    int prev_newline = 0;
    
    // If doesn't start with <USER>, add it
    if (strncmp(buffer, "<USER>", 6) != 0) {
        strcpy(dst, "<USER> ");
        dst += 7;
    }
    
    // Process the content
    while (*src) {
        if (*src == '\n') {
            if (!prev_newline) {
                *dst++ = *src;
            }
            prev_newline = 1;
        } else {
            *dst++ = *src;
            prev_newline = 0;
        }
        src++;
    }
    
    // Ensure ending with newline
    if (dst > output && *(dst-1) != '\n') {
        *dst++ = '\n';
    }
    
    *dst = '\0';
    return output;
}

// Thread function to fetch data
void *fetch_data(void *arg) {
    ThreadData *thread_data = (ThreadData *)arg;
    
    // Initialize buffer
    thread_data->buffer = malloc(thread_data->buffer_size);
    if (!thread_data->buffer) {
        pthread_mutex_lock(thread_data->console_mutex);
        fprintf(stderr, "Thread %d: Failed to allocate memory\n", thread_data->thread_id);
        pthread_mutex_unlock(thread_data->console_mutex);
        return NULL;
    }
    thread_data->buffer[0] = '\0';
    thread_data->buffer_pos = 0;
    
    CURL *curl = curl_easy_init();
    if (!curl) {
        pthread_mutex_lock(thread_data->console_mutex);
        fprintf(stderr, "Thread %d: Failed to initialize curl\n", thread_data->thread_id);
        pthread_mutex_unlock(thread_data->console_mutex);
        free(thread_data->buffer);
        thread_data->buffer = NULL;
        return NULL;
    }
    
    const char *api_key = getenv("ANTHROPIC_API_KEY");
    if (!api_key) {
        pthread_mutex_lock(thread_data->console_mutex);
        fprintf(stderr, "Thread %d: ANTHROPIC_API_KEY not set\n", thread_data->thread_id);
        pthread_mutex_unlock(thread_data->console_mutex);
        curl_easy_cleanup(curl);
        free(thread_data->buffer);
        thread_data->buffer = NULL;
        return NULL;
    }

    // Set up headers
    struct curl_slist *headers = NULL;
    char auth_header[256];
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s", api_key);
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    // Request body
    char request_body[4096];
    snprintf(request_body, sizeof(request_body), 
        "{"
        "\"model\": \"claude-3-5-sonnet-20241022\","
        "\"max_tokens\": 4096,"
        "\"stream\": true,"
        "\"messages\": [{"
        "  \"role\": \"user\","
        "  \"content\": \"%s\""
        "}]"
        "}", prompt);

    // Set up and perform request
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_body);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, thread_data);

    pthread_mutex_lock(thread_data->console_mutex);
    printf("Thread %d: Starting request\n", thread_data->thread_id);
    pthread_mutex_unlock(thread_data->console_mutex);
    
    CURLcode res = curl_easy_perform(curl);
    
    pthread_mutex_lock(thread_data->console_mutex);
    if (res != CURLE_OK) {
        fprintf(stderr, "Thread %d: curl_easy_perform() failed: %s\n", 
                thread_data->thread_id, curl_easy_strerror(res));
    } else {
        printf("Thread %d: Request completed successfully\n", thread_data->thread_id);
    }
    pthread_mutex_unlock(thread_data->console_mutex);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return NULL;
}

int main(void) {
    // Global CURL initialization
    curl_global_init(CURL_GLOBAL_ALL);
    
    printf("Starting data collection with %d threads...\n", NUM_THREADS);
    
    // Thread resources
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    
    // Initialize and create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].buffer_size = MAX_BUFFER_SIZE;
        thread_data[i].console_mutex = &console_mutex;
        
        int result = pthread_create(&threads[i], NULL, fetch_data, &thread_data[i]);
        if (result != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            curl_global_cleanup();
            return 1;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        printf("Thread %d completed\n", i);
    }
    
    printf("All threads finished. Processing and saving data...\n");
    
    // Process and write all buffers to data.txt
    FILE *output = fopen("data.txt", "a");
    if (!output) {
        fprintf(stderr, "Failed to open data.txt for writing\n");
        curl_global_cleanup();
        return 1;
    }
    
    int examples_count = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        if (thread_data[i].buffer && thread_data[i].buffer_pos > 0) {
            // Process the buffer
            char *processed = process_buffer(thread_data[i].buffer);
            if (processed) {
                // Count examples by USER tags
                const char *temp = processed;
                while ((temp = strstr(temp, "<USER>")) != NULL) {
                    examples_count++;
                    temp += 6;
                }
                
                // Write to file
                fwrite(processed, 1, strlen(processed), output);
                free(processed);
            }
            free(thread_data[i].buffer);
        }
    }
    
    fclose(output);
    printf("Data collection complete. Approximately %d examples saved to data.txt\n", examples_count);
    
    curl_global_cleanup();
    return 0;
}