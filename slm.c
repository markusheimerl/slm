#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "slm.h"

int main() {
    srand(time(NULL));
    
    // Hyperparameters
    const int embed_dim = 64;
    const int batch_size = 16;
    const int num_epochs = 100;
    const int steps_per_epoch = 1000;
    const float learning_rate = 0.0001f;
    
    printf("Initializing SLM with embed_dim=%d, batch_size=%d\n", embed_dim, batch_size);
    
    // Load text data
    TextData* text_data = load_text_data("combined_corpus.txt");
    if (!text_data) {
        fprintf(stderr, "Failed to load text data\n");
        return 1;
    }
    
    // Initialize model
    SLM* slm = init_slm(embed_dim, batch_size, text_data);
    
    printf("Starting training for %d epochs with %d steps per epoch...\n", 
           num_epochs, steps_per_epoch);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int step = 0; step < steps_per_epoch; step++) {
            float step_loss = train_step(slm, text_data, learning_rate);
            epoch_loss += step_loss;
            
            // Check for NaN or infinite loss
            if (!isfinite(step_loss)) {
                printf("Loss became non-finite at epoch %d, step %d\n", epoch + 1, step + 1);
                goto cleanup;
            }
        }
        
        epoch_loss /= steps_per_epoch;
        float perplexity = expf(epoch_loss);
        
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
    
cleanup:
    // Cleanup
    free_slm(slm);
    free_text_data(text_data);
    
    printf("\nTraining completed!\n");
    return 0;
}