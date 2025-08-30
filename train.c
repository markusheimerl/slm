#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "slm.h"
#include "data.h"

void print_sample_sequences(unsigned char* input_chars, unsigned char* target_chars, int seq_len) {
    printf("Sample sequences for inspection:\n");
    printf("==================================\n");
    for (int i = 0; i < 3; i++) {
        printf("Sample %d:\n", i);
        printf("Input:  \"");
        for (int j = 0; j < seq_len; j++) {
            char c = input_chars[i * seq_len + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '?');
        }
        printf("\"\nTarget: \"");
        for (int j = 0; j < seq_len; j++) {
            char c = target_chars[i * seq_len + j];
            printf("%c", (c >= 32 && c <= 126) ? c : '?');
        }
        printf("\"\n\n");
    }
}

void train_model(SLM* slm, unsigned char* input_chars, unsigned char* target_chars, 
                int num_sequences, int batch_size, int num_epochs, float learning_rate) {
    const int num_batches = (num_sequences + batch_size - 1) / batch_size;
    
    // Allocate device memory for input and target
    int seq_size = batch_size * slm->seq_len;
    unsigned char *d_input_chars, *d_target_chars;
    CHECK_CUDA(cudaMalloc(&d_input_chars, seq_size * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_target_chars, seq_size * sizeof(unsigned char)));
    
    printf("Starting training...\n");
    printf("Architecture: %d layers, d_model=%d, seq_len=%d, vocab_size=%d, batch_size=%d, num_sequences=%d, num_batches=%d\n\n", 
           slm->num_layers, slm->d_model, slm->seq_len, slm->vocab_size, batch_size, num_sequences, num_batches);
    
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = (start_idx + batch_size > num_sequences) ? num_sequences : start_idx + batch_size;
            if (end_idx - start_idx < batch_size) continue;
            
            unsigned char* input_batch = input_chars + start_idx * slm->seq_len;
            unsigned char* target_batch = target_chars + start_idx * slm->seq_len;
            
            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_input_chars, input_batch, seq_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_target_chars, target_batch, seq_size * sizeof(unsigned char), cudaMemcpyHostToDevice));
            
            forward_pass_slm(slm, d_input_chars);
            total_loss += calculate_loss_slm(slm, d_target_chars);

            if (epoch < num_epochs) {
                zero_gradients_slm(slm);
                backward_pass_slm(slm, d_input_chars);
                update_weights_slm(slm, learning_rate);
            }
        }

        if (epoch % 5 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f, Perplexity: %.2f\n", 
                   epoch, num_epochs, total_loss / num_batches, expf(total_loss / num_batches));
        }
    }
    
    CHECK_CUDA(cudaFree(d_input_chars));
    CHECK_CUDA(cudaFree(d_target_chars));
}

int main() {
    srand(time(NULL));

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Model parameters
    const int d_model = 128;
    const int seq_len = 64;
    const int num_layers = 4;
    const int mlp_hidden = 256;
    const int batch_size = 32;
    
    // Training parameters
    const int num_sequences = 4096;
    const int num_epochs = 50;
    const float learning_rate = 0.001f;
    
    // Load corpus
    size_t corpus_size;
    char* corpus = load_corpus("corpus.txt", &corpus_size);
    if (!corpus) {
        printf("Failed to load corpus. Creating dummy data instead.\n");
        corpus = strdup("The quick brown fox jumps over the lazy dog. ");
        corpus_size = strlen(corpus);
    }
    
    // Generate character sequences
    unsigned char* input_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    unsigned char* target_chars = (unsigned char*)malloc(num_sequences * seq_len * sizeof(unsigned char));
    
    generate_char_sequences_from_corpus(&input_chars, &target_chars, num_sequences, seq_len, corpus, corpus_size);
    print_sample_sequences(input_chars, target_chars, seq_len);
    
    // Initialize and train model
    SLM* slm = init_slm(d_model, seq_len, batch_size, mlp_hidden, num_layers, cublas_handle);
    train_model(slm, input_chars, target_chars, num_sequences, batch_size, num_epochs, learning_rate);

    // Get timestamp and save model
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_slm.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_sequences.txt", localtime(&now));

    save_slm(slm, model_fname);
    
    // Save sequences for inspection
    FILE* seq_file = fopen(data_fname, "w");
    if (seq_file) {
        for (int i = 0; i < 100; i++) {  // Save first 100 sequences
            fprintf(seq_file, "Sequence %d:\nInput:  ", i);
            for (int j = 0; j < seq_len; j++) {
                fprintf(seq_file, "%c", input_chars[i * seq_len + j]);
            }
            fprintf(seq_file, "\nTarget: ");
            for (int j = 0; j < seq_len; j++) {
                fprintf(seq_file, "%c", target_chars[i * seq_len + j]);
            }
            fprintf(seq_file, "\n\n");
        }
        fclose(seq_file);
        printf("Sequences saved to %s\n", data_fname);
    }
    
    // Generate some text
    printf("\nGenerating text...\n");
    unsigned char seed[] = "The quick brown";
    generate_text(slm, seed, strlen((char*)seed), 100, 1.0f);
    
    // Cleanup
    free(corpus);
    free(input_chars);
    free(target_chars);
    free_slm(slm);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}