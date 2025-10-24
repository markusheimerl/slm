#include "data.h"

// Load the text corpus from a file
char* load_corpus(const char* filename, size_t* corpus_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open corpus file: %s\n", filename);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    *corpus_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (*corpus_size == 0) {
        printf("Error: Corpus file is empty: %s\n", filename);
        fclose(file);
        return NULL;
    }

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

// Extract sections starting with <|bos|> and ending with <|eos|> (only sections <= seq_len)
void extract_bos_sections(char* corpus, size_t corpus_size, unsigned char** input_tokens, unsigned char** target_tokens, int* num_sections, int seq_len) {
    const char* bos_marker = "<|bos|>";
    const char* eos_marker = "<|eos|>";
    const int bos_len = 7;
    const int eos_len = 7;
    
    // First pass: count valid sections (<= seq_len)
    int valid_section_count = 0;
    
    for (size_t i = 0; i <= corpus_size - bos_len; i++) {
        if (strncmp(&corpus[i], bos_marker, bos_len) == 0) {
            size_t section_start = i;
            size_t section_end = 0;
            bool found_eos = false;
            
            // Find <|eos|> marker
            for (size_t j = i + bos_len; j <= corpus_size - eos_len; j++) {
                if (strncmp(&corpus[j], eos_marker, eos_len) == 0) {
                    section_end = j + eos_len;  // Include the <|eos|> marker
                    found_eos = true;
                    break;
                }
            }
            
            if (!found_eos) {
                // Skip sections without <|eos|>
                continue;
            }
            
            // Calculate section length
            size_t section_length = section_end - section_start;
            
            // Only count if section is short enough
            if (section_length <= (size_t)seq_len) {
                valid_section_count++;
            }
            
            // Jump to after the current section
            i = section_end - 1;
        }
    }
    
    printf("Found %d sections with length <= %d\n", valid_section_count, seq_len);
    
    if (valid_section_count == 0) {
        printf("Error: No sections <= %d characters found in corpus\n", seq_len);
        *num_sections = 0;
        return;
    }
    
    // Allocate memory for valid sections only
    *input_tokens = (unsigned char*)malloc(valid_section_count * seq_len * sizeof(unsigned char));
    *target_tokens = (unsigned char*)malloc(valid_section_count * seq_len * sizeof(unsigned char));
    
    // Second pass: extract valid sections
    int current_section = 0;
    
    for (size_t i = 0; i <= corpus_size - bos_len; i++) {
        // Find <|bos|> marker
        if (strncmp(&corpus[i], bos_marker, bos_len) == 0) {
            size_t section_start = i;
            size_t section_end = 0;
            bool found_eos = false;
            
            // Find <|eos|> marker
            for (size_t j = i + bos_len; j <= corpus_size - eos_len; j++) {
                if (strncmp(&corpus[j], eos_marker, eos_len) == 0) {
                    section_end = j + eos_len;  // Include the <|eos|> marker
                    found_eos = true;
                    break;
                }
            }
            
            if (!found_eos) {
                // Skip sections without <|eos|>
                continue;
            }
            
            // Calculate section length
            size_t section_length = section_end - section_start;
            
            // Only extract if section is short enough
            if (section_length <= (size_t)seq_len) {
                // Copy the section (from <|bos|> to <|eos|> inclusive)
                memcpy(&(*input_tokens)[current_section * seq_len], &corpus[section_start], section_length);
                
                // Pad with whitespaces after <|eos|> until seq_len
                for (size_t k = section_length; k < (size_t)seq_len; k++) {
                    (*input_tokens)[current_section * seq_len + k] = (unsigned char)' ';
                }
                
                // Create target_tokens (shifted by 1)
                memcpy(&(*target_tokens)[current_section * seq_len], 
                       &(*input_tokens)[current_section * seq_len] + 1, 
                       (seq_len - 1) * sizeof(unsigned char));
                
                // For the last target token, use a space (padding continuation)
                (*target_tokens)[current_section * seq_len + seq_len - 1] = (unsigned char)' ';
                
                current_section++;
            }
            
            if (current_section >= valid_section_count) break;
            
            // Jump to after the current section
            i = section_end - 1;
        }
    }
    
    *num_sections = current_section;
    printf("Extracted %d valid sections (seq_len=%d)\n", *num_sections, seq_len);
}