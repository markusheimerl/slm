#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

int main() {
    srand(time(NULL));
    
    // Hyperparameters
    const int embedding_dim = 128;
    const int context_length = 256;
    const int batch_size = 32;
    const int num_epochs = 100;
    const int batches_per_epoch = 100;
    const float learning_rate = 0.001f;
    
    printf("Initializing SLM with embedding_dim=%d, context_length=%d, batch_size=%d\n",
           embedding_dim, context_length, batch_size);
    
    // Load text data
    TextData* text_data = load_text_data("combined_corpus.txt");
    if (!text_data) {
        fprintf(stderr, "Failed to load text data\n");
        return 1;
    }
    
    // Initialize model
    SLM* slm = init_slm(embedding_dim, context_length, batch_size);
    
    printf("Starting training for %d epochs with %d batches per epoch...\n", 
           num_epochs, batches_per_epoch);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            float batch_loss = train_batch(slm, text_data, learning_rate);
            epoch_loss += batch_loss;
        }
        
        epoch_loss /= batches_per_epoch;
        float perplexity = expf(epoch_loss / batch_size);
        
        printf("Epoch [%d/%d], Loss: %.4f, Perplexity: %.2f\n", 
               epoch + 1, num_epochs, epoch_loss, perplexity);
        
        // Generate sample text every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            generate_text(slm, "The ", 50);
        }
    }
    
    // Final text generation
    printf("\nFinal text generation:\n");
    generate_text(slm, "Once upon a time", 200);
    generate_text(slm, "In the beginning", 200);
    generate_text(slm, "The quick brown", 200);
    
    // Cleanup
    free_slm(slm);
    free_text_data(text_data);
    
    printf("\nTraining completed successfully!\n");
    return 0;
}