#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

// Reshape data from [batch][time][feature] to [time][batch][feature]
void reshape_data_for_batch_processing(float* X, float* y, 
                                     float** X_reshaped, float** y_reshaped,
                                     int num_sequences, int seq_len, 
                                     int input_dim, int output_dim) {
    // Reshape to: seq_len tensors of size (batch_size x input_dim/output_dim)
    *X_reshaped = (float*)malloc(seq_len * num_sequences * input_dim * sizeof(float));
    *y_reshaped = (float*)malloc(seq_len * num_sequences * output_dim * sizeof(float));
    
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < num_sequences; b++) {
            // Original layout: [seq][time][feature]
            int orig_x_idx = b * seq_len * input_dim + t * input_dim;
            int orig_y_idx = b * seq_len * output_dim + t * output_dim;
            
            // New layout: [time][seq][feature] 
            int new_x_idx = t * num_sequences * input_dim + b * input_dim;
            int new_y_idx = t * num_sequences * output_dim + b * output_dim;
            
            memcpy(&(*X_reshaped)[new_x_idx], &X[orig_x_idx], input_dim * sizeof(float));
            memcpy(&(*y_reshaped)[new_y_idx], &y[orig_y_idx], output_dim * sizeof(float));
        }
    }
}

int main() {
    srand(time(NULL));
    
    // Parameters matching the SSM setup
    const int input_dim = 16;      // EMBED_DIM - embedding dimension
    const int seq_len = 64;        // Sequence length
    const int num_sequences = 64;  // Batch size (number of sequences)
    const int output_dim = 128;    // MAX_CHAR_VALUE - vocabulary size for one-hot
    
    printf("Generating text sequence data...\n");
    printf("Input dim (embedding): %d\n", input_dim);
    printf("Output dim (vocab): %d\n", output_dim);
    printf("Sequence length: %d\n", seq_len);
    printf("Number of sequences: %d\n", num_sequences);
    
    // Generate text sequence data from corpus
    float *X, *y;
    generate_text_sequence_data(&X, &y, num_sequences, seq_len, input_dim, output_dim, 
                               "combined_corpus.txt");
    
    // Reshape data for batch processing (same as SSM)
    float *X_reshaped, *y_reshaped;
    reshape_data_for_batch_processing(X, y, &X_reshaped, &y_reshaped,
                                    num_sequences, seq_len, input_dim, output_dim);
    
    // Get timestamp for filename
    char data_fname[64];
    time_t now = time(NULL);
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_text_data.csv", 
             localtime(&now));
    
    // Save data to CSV for inspection
    save_text_sequence_data_to_csv(X, y, num_sequences, seq_len, input_dim, output_dim, data_fname);
    
    printf("\nData generation complete!\n");
    printf("Original data layout: [seq][time][feature]\n");
    printf("Reshaped data layout: [time][seq][feature]\n");
    printf("Ready for SSM training with cross-entropy loss.\n");
    
    // Print some sample embeddings for verification
    printf("\nSample character embeddings:\n");
    init_embeddings(); // Ensure embeddings are initialized
    
    char sample_chars[] = {'a', 'b', 'c', ' ', '\n'};
    for (int i = 0; i < 5; i++) {
        printf("'%c' (ASCII %d): [", sample_chars[i], (int)sample_chars[i]);
        for (int j = 0; j < input_dim; j++) {
            printf("%.3f", embedding_matrix[(int)sample_chars[i]][j]);
            if (j < input_dim - 1) printf(", ");
        }
        printf("]\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_reshaped);
    free(y_reshaped);
    
    return 0;
}