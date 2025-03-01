#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

// Simplified streaming callback
size_t writefunc(void *ptr, size_t size, size_t nmemb, void *unused) {
    (void)unused; // Suppress unused parameter warning
    
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
                printf("%s", unescaped);
                fflush(stdout);
                
                free(unescaped);
                free(content);
            }
        }
    }
    
    return realsize;
}

int main(void) {
    CURL *curl = curl_easy_init();
    if (!curl) return 1;
    
    const char *api_key = getenv("ANTHROPIC_API_KEY");
    if (!api_key) {
        fprintf(stderr, "ANTHROPIC_API_KEY not set\n");
        return 1;
    }

    // Set up headers
    struct curl_slist *headers = NULL;
    char auth_header[256];
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s", api_key);
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    // Request body
    const char *data = "{"
        "\"model\": \"claude-3-5-sonnet-20241022\","
        "\"max_tokens\": 4096,"
        "\"stream\": true,"
        "\"messages\": [{"
        "  \"role\": \"user\","
        "  \"content\": \"Create a training dataset of multi-turn conversations about stories. Format each example as a single line with <USER> and <ASSISTANT> tags, using \\\\n to represent newlines within stories. Separate different training examples with ONLY A SINGLE newline. Make topics diverse and CREATIVE! Also make the user ask questions about the story within a training example. Create at least 10 different multi-turn examples.\""
        "}]"
    "}";

    // Set up and perform request
    curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    
    return 0;
}