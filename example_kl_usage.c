/**
 * Example usage of KL divergence penalty in SLM training
 * 
 * This demonstrates how to:
 * 1. Initialize a model with default KL penalty (0.01)
 * 2. Adjust the KL penalty weight during training
 * 3. Use the modified loss function
 */

#include "slm.h"
#include <stdio.h>

int main() {
    // Initialize model parameters
    const int embed_dim = 128;
    const int state_dim = 64;
    const int seq_len = 512;
    const int batch_size = 32;
    
    printf("=== KL Divergence Penalty Demo ===\n\n");
    
    // 1. Initialize SLM with default KL penalty weight (0.01)
    printf("1. Initializing SLM...\n");
    SLM* slm = init_slm(embed_dim, state_dim, seq_len, batch_size);
    printf("   Default KL penalty weight: %.4f\n\n", slm->kl_penalty_weight);
    
    // 2. Demonstrate adjusting KL penalty weight
    printf("2. Adjusting KL penalty weights:\n");
    
    // Higher regularization (more uniform distributions)
    set_kl_penalty_weight_slm(slm, 0.05f);
    
    // Lower regularization (allows sharper distributions) 
    set_kl_penalty_weight_slm(slm, 0.001f);
    
    // Disable KL penalty (standard cross-entropy only)
    set_kl_penalty_weight_slm(slm, 0.0f);
    
    // Restore to reasonable value for training
    set_kl_penalty_weight_slm(slm, 0.01f);
    
    printf("\n3. Training with KL penalty:\n");
    printf("   During training, calculate_loss_slm() will return:\n");
    printf("   loss = cross_entropy_loss + %.4f * kl_divergence_loss\n", slm->kl_penalty_weight);
    
    printf("\n4. The gradient calculation automatically includes:\n");
    printf("   - Cross-entropy gradient: P(c) - 1_{target}(c)\n");
    printf("   - KL divergence gradient: λ_KL * P(c) * (log(P(c)) + 1)\n");
    
    printf("\n5. Model persistence:\n");
    printf("   save_slm() will save the KL penalty weight\n");
    printf("   load_slm() will restore the KL penalty weight\n");
    
    // Cleanup
    free_slm(slm);
    
    printf("\n=== Demo Complete ===\n");
    return 0;
}