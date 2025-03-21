#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <curl/curl.h>

#define MAX_CORPUS_SIZE 20000000  // 20MB max for combined corpus
#define MAX_FILENAME_LEN 256
#define MAX_URL_LEN 256
#define MAX_BOOKS 15
#define BATCH_SIZE 64
#define TEMPERATURE 0.7f

// Model configuration
typedef struct {
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    int num_layers;
    int seq_length;
} ModelConfig;

// Memory buffer for curl
typedef struct {
    char *memory;
    size_t size;
} MemoryStruct;

// Model structure
typedef struct {
    // Embedding layer
    float *embedding_weight;  // [vocab_size, embed_dim]
    
    // Mixer blocks
    float **token_mixing_weight1;  // [num_layers][seq_length, seq_length]
    float **token_mixing_weight2;  // [num_layers][seq_length, seq_length]
    float **channel_mixing_weight1;  // [num_layers][hidden_dim, embed_dim]
    float **channel_mixing_weight2;  // [num_layers][embed_dim, hidden_dim]
    
    // Output projection
    float *out_proj_weight;  // [vocab_size, embed_dim]
    
    // Causal mask for token mixing
    float *causal_mask;  // [seq_length, seq_length]
    
    // Temporary buffers for forward pass
    float *temp_buffer1;  // [BATCH_SIZE * seq_length * embed_dim] 
    float *temp_buffer2;  // [BATCH_SIZE * seq_length * embed_dim]
    
    ModelConfig config;
} Model;

// Dataset structure
typedef struct {
    unsigned char *data;
    size_t size;
    int seq_length;
} Dataset;

// Write callback for curl
static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    MemoryStruct *mem = (MemoryStruct *)userp;
    
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(!ptr) {
        printf("Not enough memory (realloc returned NULL)\n");
        return 0;
    }
    
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}

// Download a file using curl
char* download_file(const char *url) {
    CURL *curl_handle;
    CURLcode res;
    MemoryStruct chunk;
    
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle = curl_easy_init();
    
    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    
    res = curl_easy_perform(curl_handle);
    
    if(res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        free(chunk.memory);
        curl_easy_cleanup(curl_handle);
        curl_global_cleanup();
        return NULL;
    }
    
    curl_easy_cleanup(curl_handle);
    curl_global_cleanup();
    
    return chunk.memory;
}

// Load text file or download from URL
char* load_or_download_text(const char *url, const char *save_path) {
    FILE *f = fopen(save_path, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        char *text = malloc(fsize + 1);
        fread(text, fsize, 1, f);
        fclose(f);
        text[fsize] = 0;
        
        return text;
    }
    
    // Download the file
    char *text = download_file(url);
    if (!text) {
        fprintf(stderr, "Failed to download file from %s\n", url);
        return NULL;
    }
    
    // Save it locally
    f = fopen(save_path, "w");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
    
    return text;
}

// Get combined corpus from multiple books
char* get_combined_corpus() {
    const char *urls[MAX_BOOKS] = {
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "https://www.gutenberg.org/files/84/84-0.txt",
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "https://www.gutenberg.org/files/2701/2701-0.txt",
        "https://www.gutenberg.org/files/98/98-0.txt",
        "https://www.gutenberg.org/files/1400/1400-0.txt",
        "https://www.gutenberg.org/files/345/345-0.txt",
        "https://www.gutenberg.org/files/174/174-0.txt",
        "https://www.gutenberg.org/files/16/16-0.txt",
        "https://www.gutenberg.org/files/768/768-0.txt",
        "https://www.gutenberg.org/files/45/45-0.txt",
        "https://www.gutenberg.org/files/1260/1260-0.txt"
    };
    
    const char *filenames[MAX_BOOKS] = {
        "pride_and_prejudice.txt",
        "frankenstein.txt",
        "sherlock_holmes.txt",
        "moby_dick.txt",
        "tale_of_two_cities.txt",
        "great_expectations.txt",
        "dracula.txt",
        "dorian_gray.txt",
        "peter_pan.txt",
        "wuthering_heights.txt",
        "anne_of_green_gables.txt",
        "jane_eyre.txt"
    };
    
    int num_books = 12;
    char *combined_text = malloc(MAX_CORPUS_SIZE);
    combined_text[0] = '\0';
    size_t combined_length = 0;
    
    for (int i = 0; i < num_books; i++) {
        char filepath[MAX_FILENAME_LEN];
        sprintf(filepath, "gutenberg_texts/%s", filenames[i]);
        
        printf("Processing %s...\n", filenames[i]);
        char *text = load_or_download_text(urls[i], filepath);
        if (text) {
            size_t text_len = strlen(text);
            if (combined_length + text_len + 3 < MAX_CORPUS_SIZE) {
                strcat(combined_text, text);
                strcat(combined_text, "\n\n");
                combined_length += text_len + 2;
            } else {
                printf("Warning: Combined text exceeds max size, truncating\n");
            }
            free(text);
        }
    }
    
    printf("Combined corpus size: %zu characters\n", combined_length);
    
    // Save combined corpus
    FILE *f = fopen("gutenberg_texts/combined_corpus.txt", "w");
    if (f) {
        fputs(combined_text, f);
        fclose(f);
    }
    
    return combined_text;
}

// Encode text into byte-level tokens
unsigned char* encode_text(const char *text, size_t *size) {
    *size = strlen(text);
    unsigned char *tokens = malloc(*size);
    memcpy(tokens, text, *size);
    return tokens;
}

// Decode tokens back to text
char* decode_tokens(const unsigned char *tokens, size_t size) {
    char *text = malloc(size + 1);
    memcpy(text, tokens, size);
    text[size] = '\0';
    return text;
}

// Create dataset from text
Dataset create_dataset(const char *text, int seq_length) {
    Dataset dataset;
    dataset.seq_length = seq_length;
    dataset.data = encode_text(text, &dataset.size);
    return dataset;
}

// Free dataset memory
void free_dataset(Dataset *dataset) {
    free(dataset->data);
}

// Initialize model parameters with random values
void init_model(Model *model, ModelConfig config) {
    model->config = config;
    int vocab_size = config.vocab_size;
    int embed_dim = config.embed_dim;
    int hidden_dim = config.hidden_dim;
    int num_layers = config.num_layers;
    int seq_length = config.seq_length;
    
    // Initialize random seed
    srand(time(NULL));
    
    // Allocate memory for model parameters
    model->embedding_weight = malloc(vocab_size * embed_dim * sizeof(float));
    model->out_proj_weight = malloc(vocab_size * embed_dim * sizeof(float));
    model->causal_mask = malloc(seq_length * seq_length * sizeof(float));
    
    model->token_mixing_weight1 = malloc(num_layers * sizeof(float*));
    model->token_mixing_weight2 = malloc(num_layers * sizeof(float*));
    model->channel_mixing_weight1 = malloc(num_layers * sizeof(float*));
    model->channel_mixing_weight2 = malloc(num_layers * sizeof(float*));
    
    for (int i = 0; i < num_layers; i++) {
        model->token_mixing_weight1[i] = malloc(seq_length * seq_length * sizeof(float));
        model->token_mixing_weight2[i] = malloc(seq_length * seq_length * sizeof(float));
        model->channel_mixing_weight1[i] = malloc(hidden_dim * embed_dim * sizeof(float));
        model->channel_mixing_weight2[i] = malloc(embed_dim * hidden_dim * sizeof(float));
    }
    
    // Allocate temporary buffers
    model->temp_buffer1 = malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    model->temp_buffer2 = malloc(BATCH_SIZE * seq_length * embed_dim * sizeof(float));
    
    // Initialize embedding weights
    float embed_scale = 1.0f / sqrtf(embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        model->embedding_weight[i] = (((float)rand() / RAND_MAX) * 2 - 1) * embed_scale;
    }
    
    // Initialize causal mask (lower triangular matrix)
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            model->causal_mask[i * seq_length + j] = (j <= i) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize output projection weights
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        model->out_proj_weight[i] = (((float)rand() / RAND_MAX) * 2 - 1) * embed_scale;
    }
    
    // Initialize mixer block weights
    float seq_scale = 1.0f / sqrtf(seq_length);
    float hidden_scale = 1.0f / sqrtf(hidden_dim);
    
    for (int l = 0; l < num_layers; l++) {
        // Token mixing weights
        for (int i = 0; i < seq_length * seq_length; i++) {
            model->token_mixing_weight1[l][i] = (((float)rand() / RAND_MAX) * 2 - 1) * seq_scale;
            model->token_mixing_weight2[l][i] = (((float)rand() / RAND_MAX) * 2 - 1) * seq_scale;
        }
        
        // Channel mixing weights
        for (int i = 0; i < hidden_dim * embed_dim; i++) {
            model->channel_mixing_weight1[l][i] = (((float)rand() / RAND_MAX) * 2 - 1) * embed_scale;
        }
        
        for (int i = 0; i < embed_dim * hidden_dim; i++) {
            model->channel_mixing_weight2[l][i] = (((float)rand() / RAND_MAX) * 2 - 1) * hidden_scale;
        }
    }
}

// Free model memory
void free_model(Model *model) {
    free(model->embedding_weight);
    free(model->out_proj_weight);
    free(model->causal_mask);
    
    for (int i = 0; i < model->config.num_layers; i++) {
        free(model->token_mixing_weight1[i]);
        free(model->token_mixing_weight2[i]);
        free(model->channel_mixing_weight1[i]);
        free(model->channel_mixing_weight2[i]);
    }
    
    free(model->token_mixing_weight1);
    free(model->token_mixing_weight2);
    free(model->channel_mixing_weight1);
    free(model->channel_mixing_weight2);
    
    free(model->temp_buffer1);
    free(model->temp_buffer2);
}

// Sigmoid activation function
void sigmoid_activation(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// Element-wise multiplication
void element_wise_mul(float *a, float *b, float *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

// Apply causal mask to weights
void apply_causal_mask(float *weight, float *mask, float *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = weight[i] * mask[i];
    }
}

// Forward pass for the embedding layer
void embedding_forward(Model *model, int *input, float *output, int batch_size, int seq_len) {
    int embed_dim = model->config.embed_dim;
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token = input[b * seq_len + s];
            if (token < 0) token = 0;
            if (token >= model->config.vocab_size) token = model->config.vocab_size - 1;
            
            for (int e = 0; e < embed_dim; e++) {
                output[b * seq_len * embed_dim + s * embed_dim + e] = 
                    model->embedding_weight[token * embed_dim + e];
            }
        }
    }
}

// Forward pass for a single mixer block
void mixer_block_forward(Model *model, float *x, float *output, int batch_size, int layer_idx) {
    int seq_length = model->config.seq_length;
    int embed_dim = model->config.embed_dim;
    int hidden_dim = model->config.hidden_dim;
    
    float *temp_buffer1 = model->temp_buffer1;
    float *temp_buffer2 = model->temp_buffer2;
    
    // Copy input to output for residual connection
    memcpy(output, x, batch_size * seq_length * embed_dim * sizeof(float));
    
    // Token mixing
    // Transpose x from [batch, seq, embed] to [batch, embed, seq]
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int e = 0; e < embed_dim; e++) {
                temp_buffer1[b * embed_dim * seq_length + e * seq_length + s] = 
                    x[b * seq_length * embed_dim + s * embed_dim + e];
            }
        }
    }
    
    // Prepare masked weights for token mixing
    float *masked_weight1 = malloc(seq_length * seq_length * sizeof(float));
    float *masked_weight2 = malloc(seq_length * seq_length * sizeof(float));
    
    apply_causal_mask(model->token_mixing_weight1[layer_idx], model->causal_mask, 
                    masked_weight1, seq_length * seq_length);
    apply_causal_mask(model->token_mixing_weight2[layer_idx], model->causal_mask, 
                    masked_weight2, seq_length * seq_length);
    
    // Token mixing MLP - first layer
    for (int b = 0; b < batch_size; b++) {
        for (int e = 0; e < embed_dim; e++) {
            float *input_slice = &temp_buffer1[b * embed_dim * seq_length + e * seq_length];
            float *output_slice = &temp_buffer2[b * embed_dim * seq_length + e * seq_length];
            
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        seq_length, seq_length,
                        1.0f, masked_weight1, seq_length,
                        input_slice, 1, 0.0f, output_slice, 1);
            
            // Apply sigmoid activation
            sigmoid_activation(output_slice, seq_length);
            
            // Element-wise multiply with activation
            element_wise_mul(output_slice, output_slice, output_slice, seq_length);
            
            // Second layer
            float *temp = malloc(seq_length * sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        seq_length, seq_length,
                        1.0f, masked_weight2, seq_length,
                        output_slice, 1, 0.0f, temp, 1);
            
            // Copy result back
            memcpy(output_slice, temp, seq_length * sizeof(float));
            free(temp);
        }
    }
    
    // Transpose back from [batch, embed, seq] to [batch, seq, embed]
    for (int b = 0; b < batch_size; b++) {
        for (int e = 0; e < embed_dim; e++) {
            for (int s = 0; s < seq_length; s++) {
                temp_buffer1[b * seq_length * embed_dim + s * embed_dim + e] = 
                    temp_buffer2[b * embed_dim * seq_length + e * seq_length + s];
            }
        }
    }
    
    // Add residual connection
    for (int i = 0; i < batch_size * seq_length * embed_dim; i++) {
        output[i] += temp_buffer1[i];
    }
    
    // Store result for next residual connection
    memcpy(temp_buffer1, output, batch_size * seq_length * embed_dim * sizeof(float));
    
    // Channel mixing MLP
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            float *input_slice = &output[b * seq_length * embed_dim + s * embed_dim];
            float *output_slice = &temp_buffer2[b * seq_length * embed_dim + s * embed_dim];
            
            // First layer
            float *hidden_activations = malloc(hidden_dim * sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        hidden_dim, embed_dim,
                        1.0f, model->channel_mixing_weight1[layer_idx], embed_dim,
                        input_slice, 1, 0.0f, hidden_activations, 1);
            
            // Apply sigmoid activation
            sigmoid_activation(hidden_activations, hidden_dim);
            
            // Element-wise multiply with activation
            element_wise_mul(hidden_activations, hidden_activations, hidden_activations, hidden_dim);
            
            // Second layer
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        embed_dim, hidden_dim,
                        1.0f, model->channel_mixing_weight2[layer_idx], hidden_dim,
                        hidden_activations, 1, 0.0f, output_slice, 1);
            
            free(hidden_activations);
        }
    }
    
    // Add residual connection
    for (int i = 0; i < batch_size * seq_length * embed_dim; i++) {
        output[i] = temp_buffer1[i] + temp_buffer2[i];
    }
    
    free(masked_weight1);
    free(masked_weight2);
}

// Forward pass for the entire model
void model_forward(Model *model, int *input, float *output, int batch_size) {
    int seq_length = model->config.seq_length;
    int embed_dim = model->config.embed_dim;
    
    float *embeddings = malloc(batch_size * seq_length * embed_dim * sizeof(float));
    float *block_output = malloc(batch_size * seq_length * embed_dim * sizeof(float));
    
    // Embedding layer
    embedding_forward(model, input, embeddings, batch_size, seq_length);
    
    // Process through mixer blocks
    float *block_input = embeddings;
    for (int l = 0; l < model->config.num_layers; l++) {
        mixer_block_forward(model, block_input, block_output, batch_size, l);
        block_input = block_output;
    }
    
    // Final output projection
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            float *embed_vec = &block_output[b * seq_length * embed_dim + s * embed_dim];
            float *logits = &output[b * seq_length * model->config.vocab_size + s * model->config.vocab_size];
            
            cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                        model->config.vocab_size, embed_dim,
                        1.0f, model->out_proj_weight, embed_dim,
                        embed_vec, 1, 0.0f, logits, 1);
        }
    }
    
    free(embeddings);
    free(block_output);
}

// Softmax function
void softmax(float *logits, float *probs, int size) {
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf((logits[i] - max_val) / TEMPERATURE);
        sum += probs[i];
    }
    
    for (int i = 0; i < size; i++) {
        probs[i] /= sum;
    }
}

// Sample from probability distribution
int sample_from_probs(float *probs, int size) {
    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    
    for (int i = 0; i < size; i++) {
        cdf += probs[i];
        if (r < cdf) {
            return i;
        }
    }
    
    return size - 1;  // Fallback
}

// Generate text from model
char* generate_text(Model *model, const char *seed_text, int max_length) {
    int seq_length = model->config.seq_length;
    int vocab_size = model->config.vocab_size;
    
    // Encode seed text
    size_t seed_size;
    unsigned char *seed_tokens = encode_text(seed_text, &seed_size);
    
    // Prepare input sequence
    int *input_sequence = malloc(seq_length * sizeof(int));
    memset(input_sequence, 0, seq_length * sizeof(int));
    
    // Copy as much of the seed as will fit
    int copy_len = (seed_size < seq_length) ? seed_size : seq_length;
    int start_pos = (seed_size < seq_length) ? 0 : (seed_size - seq_length);
    
    for (int i = 0; i < copy_len; i++) {
        input_sequence[seq_length - copy_len + i] = seed_tokens[start_pos + i];
    }
    
    // Allocate memory for generated text
    unsigned char *generated_tokens = malloc((copy_len + max_length) * sizeof(unsigned char));
    memcpy(generated_tokens, &seed_tokens[start_pos], copy_len * sizeof(unsigned char));
    
    // Prepare output buffer
    float *output_logits = malloc(seq_length * vocab_size * sizeof(float));
    float *next_token_probs = malloc(vocab_size * sizeof(float));
    
    // Generate tokens
    for (int i = 0; i < max_length; i++) {
        // Forward pass
        model_forward(model, input_sequence, output_logits, 1);
        
        // Get logits for the last token
        float *last_token_logits = &output_logits[(seq_length - 1) * vocab_size];
        
        // Apply softmax to get probabilities
        softmax(last_token_logits, next_token_probs, vocab_size);
        
        // Sample next token
        int next_token = sample_from_probs(next_token_probs, vocab_size);
        
        // Add to generated sequence
        generated_tokens[copy_len + i] = (unsigned char)next_token;
        
        // Shift input sequence
        for (int j = 0; j < seq_length - 1; j++) {
            input_sequence[j] = input_sequence[j + 1];
        }
        input_sequence[seq_length - 1] = next_token;
    }
    
    // Decode the generated text
    char *generated_text = decode_tokens(generated_tokens, copy_len + max_length);
    
    // Clean up
    free(seed_tokens);
    free(input_sequence);
    free(generated_tokens);
    free(output_logits);
    free(next_token_probs);
    
    return generated_text;
}

// Cross entropy loss function
float cross_entropy_loss(float *logits, int *targets, int batch_size, int seq_length, int vocab_size) {
    float total_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            int target = targets[b * seq_length + s];
            float *current_logits = &logits[b * seq_length * vocab_size + s * vocab_size];
            
            // Compute softmax
            float max_val = current_logits[0];
            for (int i = 1; i < vocab_size; i++) {
                if (current_logits[i] > max_val) {
                    max_val = current_logits[i];
                }
            }
            
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum += expf(current_logits[i] - max_val);
            }
            
            float log_sum = logf(sum) + max_val;
            float correct_logit = current_logits[target];
            
            // Cross entropy loss
            total_loss += log_sum - correct_logit;
        }
    }
    
    return total_loss / (batch_size * seq_length);
}

// Simple SGD optimizer update
void sgd_update(Model *model, float learning_rate) {
    // This is a very simplified SGD update
    // A real implementation would compute gradients and update accordingly
    // This is just a placeholder to show where parameter updates would happen
    printf("SGD update would happen here with learning rate %f\n", learning_rate);
}

// Function to get a batch from the dataset
void get_batch(Dataset *dataset, int *x_batch, int *y_batch, int batch_size, int seq_length) {
    for (int b = 0; b < batch_size; b++) {
        // Random starting position for each sequence in the batch
        int start_idx = rand() % (dataset->size - seq_length - 1);
        
        for (int s = 0; s < seq_length; s++) {
            x_batch[b * seq_length + s] = dataset->data[start_idx + s];
            y_batch[b * seq_length + s] = dataset->data[start_idx + s + 1];
        }
    }
}

// Training loop
void train_model(Model *model, Dataset *dataset, int epochs, float learning_rate) {
    int batch_size = BATCH_SIZE;
    int seq_length = model->config.seq_length;
    int vocab_size = model->config.vocab_size;
    
    // Allocate memory for batches
    int *x_batch = malloc(batch_size * seq_length * sizeof(int));
    int *y_batch = malloc(batch_size * seq_length * sizeof(int));
    float *output_logits = malloc(batch_size * seq_length * vocab_size * sizeof(float));
    
    // Number of batches per epoch (approximate)
    int batches_per_epoch = dataset->size / (batch_size * seq_length);
    if (batches_per_epoch < 1) batches_per_epoch = 1;
    
    printf("Training with %d batches per epoch for %d epochs\n", batches_per_epoch, epochs);
    
    time_t start_time = time(NULL);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            // Get a random batch
            get_batch(dataset, x_batch, y_batch, batch_size, seq_length);
            
            // Forward pass
            model_forward(model, x_batch, output_logits, batch_size);
            
            // Compute loss
            float loss = cross_entropy_loss(output_logits, y_batch, batch_size, seq_length, vocab_size);
            epoch_loss += loss;
            
            // Update parameters
            sgd_update(model, learning_rate);
            
            // Print progress
            if (batch % 10 == 0) {
                time_t current_time = time(NULL);
                printf("Epoch %d/%d, Batch %d/%d, Loss: %.4f, Time: %ld seconds\n", 
                       epoch + 1, epochs, batch + 1, batches_per_epoch, loss, current_time - start_time);
                
                // Generate some text periodically
                if (batch % 50 == 0) {
                    printf("\n--- GENERATING TEXT ---\n");
                    char *seed_text = "Once upon a time";
                    char *generated = generate_text(model, seed_text, 100);
                    printf("Seed: \"%s\"\n", seed_text);
                    printf("Generated: \"%s\"\n\n", generated);
                    free(generated);
                }
            }
        }
        
        printf("Epoch %d/%d completed, Average Loss: %.4f\n", 
               epoch + 1, epochs, epoch_loss / batches_per_epoch);
    }
    
    free(x_batch);
    free(y_batch);
    free(output_logits);
}

// Count model parameters
int count_parameters(Model *model) {
    int total = 0;
    
    // Embedding layer
    total += model->config.vocab_size * model->config.embed_dim;
    
    // Mixer blocks
    for (int i = 0; i < model->config.num_layers; i++) {
        // Token mixing
        total += 2 * model->config.seq_length * model->config.seq_length;
        
        // Channel mixing
        total += model->config.hidden_dim * model->config.embed_dim;
        total += model->config.embed_dim * model->config.hidden_dim;
    }
    
    // Output projection
    total += model->config.vocab_size * model->config.embed_dim;
    
    printf("Total parameters: %d\n", total);
    return total;
}

int main() {
    // Download corpus
    char *corpus = get_combined_corpus();
    if (!corpus) {
        fprintf(stderr, "Failed to load corpus\n");
        return 1;
    }
    
    // Model configuration
    ModelConfig config = {
        .vocab_size = 256,           // Byte-level tokenization
        .embed_dim = 16,             // Embedding dimension
        .hidden_dim = 512,           // Hidden dimension
        .num_layers = 4,             // Number of mixer layers
        .seq_length = 1024           // Sequence length
    };
    
    // Initialize model
    Model model;
    init_model(&model, config);
    
    // Count parameters
    count_parameters(&model);
    
    // Create dataset
    Dataset dataset = create_dataset(corpus, config.seq_length);
    
    // Train model
    train_model(&model, &dataset, 3, 0.0001f);
    
    // Generate text from a seed
    const char *seed_text = "The quick brown fox jumps over the lazy dog";
    char *generated_text = generate_text(&model, seed_text, 100);
    printf("\nFinal generation:\n");
    printf("Seed: \"%s\"\n", seed_text);
    printf("Generated: \"%s\"\n", generated_text);
    
    // Clean up
    free(corpus);
    free(generated_text);
    free_dataset(&dataset);
    free_model(&model);
    
    return 0;
}