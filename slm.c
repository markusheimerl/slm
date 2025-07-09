#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "ssm/gpu/ssm.h"

#define VOCAB_SIZE 256  // For byte-level modeling
#define EMBEDDING_DIM 128
#define STATE_DIM 256
#define SEQ_LEN 128
#define BATCH_SIZE 64

typedef struct {
    // Device pointers for embeddings
    float* d_embeddings;        // vocab_size x embedding_dim
    float* d_output_weights;    // embedding_dim x vocab_size
    
    // Gradients
    float* d_embed_grad;        // vocab_size x embedding_dim
    float* d_output_grad;       // embedding_dim x vocab_size
    
    // Adam optimizer state
    float* d_embed_m;
    float* d_embed_v;
    float* d_output_m;
    float* d_output_v;
    
    // Dimensions
    int vocab_size;
    int embedding_dim;
    
    // cuBLAS handle (shared with SSM)
    cublasHandle_t cublas_handle;
} LanguageModel;

// CUDA kernel for embedding lookup
__global__ void embedding_forward_kernel(float* output, float* embeddings, int* indices,
                                       int seq_len, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        output[idx] = embeddings[token_id * embedding_dim + dim];
    }
}

// CUDA kernel for embedding backward
__global__ void embedding_backward_kernel(float* embed_grad, float* output_grad, int* indices,
                                        int seq_len, int batch_size, int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * batch_size * embedding_dim;
    
    if (idx < total_elements) {
        int pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int token_id = indices[pos];
        
        atomicAdd(&embed_grad[token_id * embedding_dim + dim], output_grad[idx]);
    }
}

// CUDA kernel for cross-entropy loss forward
__global__ void cross_entropy_forward_kernel(float* loss, float* grad, float* logits, int* targets,
                                           int seq_len, int batch_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = seq_len * batch_size;
    
    if (idx < total_positions) {
        float* logit_ptr = logits + idx * vocab_size;
        int target = targets[idx];
        
        // Compute max for numerical stability
        float max_logit = logit_ptr[0];
        for (int i = 1; i < vocab_size; i++) {
            max_logit = fmaxf(max_logit, logit_ptr[i]);
        }
        
        // Compute softmax denominator
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sum_exp += expf(logit_ptr[i] - max_logit);
        }
        
        // Compute loss for this position
        float position_loss = -logit_ptr[target] + max_logit + logf(sum_exp);
        atomicAdd(loss, position_loss);
        
        // Compute gradient
        float* grad_ptr = grad + idx * vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            float prob = expf(logit_ptr[i] - max_logit) / sum_exp;
            grad_ptr[i] = prob;
            if (i == target) {
                grad_ptr[i] -= 1.0f;
            }
        }
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel(float* param, float* grad, float* m, float* v,
                                  float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // Bias correction
        float bias_correction1 = 1.0f - powf(beta1, t);
        float bias_correction2 = 1.0f - powf(beta2, t);
        
        // Update biased first and second moment
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // Compute bias-corrected learning rate
        float step_size = lr * sqrtf(bias_correction2) / bias_correction1;
        
        // AdamW update with weight decay
        param[idx] = param[idx] * (1.0f - lr * weight_decay) - step_size * m[idx] / (sqrtf(v[idx]) + eps);
    }
}

// Initialize language model
LanguageModel* init_language_model(int vocab_size, int embedding_dim, cublasHandle_t cublas_handle) {
    LanguageModel* lm = (LanguageModel*)malloc(sizeof(LanguageModel));
    lm->vocab_size = vocab_size;
    lm->embedding_dim = embedding_dim;
    lm->cublas_handle = cublas_handle;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&lm->d_embeddings, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_output_weights, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_embed_grad, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_output_grad, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_embed_m, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_embed_v, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_output_m, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&lm->d_output_v, embedding_dim * vocab_size * sizeof(float)));
    
    // Initialize weights on host then copy
    float* h_embeddings = (float*)malloc(vocab_size * embedding_dim * sizeof(float));
    float* h_output_weights = (float*)malloc(embedding_dim * vocab_size * sizeof(float));
    
    // Xavier initialization
    float embed_scale = sqrtf(2.0f / (vocab_size + embedding_dim));
    float output_scale = sqrtf(2.0f / (embedding_dim + vocab_size));
    
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        h_embeddings[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * embed_scale;
    }
    for (int i = 0; i < embedding_dim * vocab_size; i++) {
        h_output_weights[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * output_scale;
    }
    
    CHECK_CUDA(cudaMemcpy(lm->d_embeddings, h_embeddings, vocab_size * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(lm->d_output_weights, h_output_weights, embedding_dim * vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Zero Adam states
    CHECK_CUDA(cudaMemset(lm->d_embed_m, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(lm->d_embed_v, 0, vocab_size * embedding_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(lm->d_output_m, 0, embedding_dim * vocab_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(lm->d_output_v, 0, embedding_dim * vocab_size * sizeof(float)));
    
    free(h_embeddings);
    free(h_output_weights);
    
    return lm;
}

// Free language model
void free_language_model(LanguageModel* lm) {
    cudaFree(lm->d_embeddings);
    cudaFree(lm->d_output_weights);
    cudaFree(lm->d_embed_grad);
    cudaFree(lm->d_output_grad);
    cudaFree(lm->d_embed_m);
    cudaFree(lm->d_embed_v);
    cudaFree(lm->d_output_m);
    cudaFree(lm->d_output_v);
    free(lm);
}

// Load text and prepare sequences
void load_text_sequences(const char* filename, int** d_input_ids, int** d_target_ids, 
                        int* num_batches, int seq_len, int batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open %s\n", filename);
        exit(1);
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read text
    unsigned char* text = (unsigned char*)malloc(file_size);
    fread(text, 1, file_size, file);
    fclose(file);
    
    // Calculate number of sequences
    int total_sequences = (file_size - 1) / seq_len;
    *num_batches = total_sequences / batch_size;
    int used_sequences = (*num_batches) * batch_size;
    
    printf("Loaded %ld bytes, creating %d batches of %d sequences\n", 
           file_size, *num_batches, batch_size);
    
    // Prepare sequences on host
    int* h_input_ids = (int*)malloc(used_sequences * seq_len * sizeof(int));
    int* h_target_ids = (int*)malloc(used_sequences * seq_len * sizeof(int));
    
    for (int seq = 0; seq < used_sequences; seq++) {
        int start_idx = seq * seq_len;
        for (int pos = 0; pos < seq_len; pos++) {
            h_input_ids[seq * seq_len + pos] = text[start_idx + pos];
            h_target_ids[seq * seq_len + pos] = text[start_idx + pos + 1];
        }
    }
    
    // Allocate and copy to device
    CHECK_CUDA(cudaMalloc(d_input_ids, used_sequences * seq_len * sizeof(int)));
    CHECK_CUDA(cudaMalloc(d_target_ids, used_sequences * seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(*d_input_ids, h_input_ids, used_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_target_ids, h_target_ids, used_sequences * seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    free(text);
    free(h_input_ids);
    free(h_target_ids);
}

// Training step
float train_step(LanguageModel* lm, SSM* ssm, int* d_input_batch, int* d_target_batch,
                float learning_rate, float weight_decay, int adam_t) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Allocate workspace
    float* d_embedded;
    float* d_logits;
    float* d_loss;
    float* d_logit_grad;
    
    CHECK_CUDA(cudaMalloc(&d_embedded, SEQ_LEN * BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logit_grad, SEQ_LEN * BATCH_SIZE * VOCAB_SIZE * sizeof(float)));
    
    // Clear gradients
    CHECK_CUDA(cudaMemset(lm->d_embed_grad, 0, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(lm->d_output_grad, 0, EMBEDDING_DIM * VOCAB_SIZE * sizeof(float)));
    zero_gradients_ssm(ssm);
    
    // Forward pass
    // 1. Embedding lookup
    int block_size = 256;
    int num_blocks = (SEQ_LEN * BATCH_SIZE * EMBEDDING_DIM + block_size - 1) / block_size;
    embedding_forward_kernel<<<num_blocks, block_size>>>(
        d_embedded, lm->d_embeddings, d_input_batch,
        SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM
    );
    
    // 2. SSM forward
    forward_pass_ssm(ssm, d_embedded);
    
    // 3. Output projection
    CHECK_CUBLAS(cublasSgemm(lm->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            VOCAB_SIZE, SEQ_LEN * BATCH_SIZE, EMBEDDING_DIM,
                            &alpha, lm->d_output_weights, EMBEDDING_DIM,
                            ssm->d_predictions, EMBEDDING_DIM,
                            &beta, d_logits, VOCAB_SIZE));
    
    // 4. Cross-entropy loss
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    num_blocks = (SEQ_LEN * BATCH_SIZE + block_size - 1) / block_size;
    cross_entropy_forward_kernel<<<num_blocks, block_size>>>(
        d_loss, d_logit_grad, d_logits, d_target_batch,
        SEQ_LEN, BATCH_SIZE, VOCAB_SIZE
    );
    
    // Get loss value
    float loss_val;
    CHECK_CUDA(cudaMemcpy(&loss_val, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    loss_val /= (SEQ_LEN * BATCH_SIZE);
    
    // Backward pass
    // 1. Output weight gradient
    CHECK_CUBLAS(cublasSgemm(lm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            EMBEDDING_DIM, VOCAB_SIZE, SEQ_LEN * BATCH_SIZE,
                            &alpha, ssm->d_predictions, EMBEDDING_DIM,
                            d_logit_grad, VOCAB_SIZE,
                            &alpha, lm->d_output_grad, EMBEDDING_DIM));
    
    // 2. SSM output gradient
    CHECK_CUBLAS(cublasSgemm(lm->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            EMBEDDING_DIM, SEQ_LEN * BATCH_SIZE, VOCAB_SIZE,
                            &alpha, lm->d_output_weights, EMBEDDING_DIM,
                            d_logit_grad, VOCAB_SIZE,
                            &beta, ssm->d_error, EMBEDDING_DIM));
    
    // 3. SSM backward
    backward_pass_ssm(ssm, d_embedded);
    
    // 4. Embedding gradient (ssm->d_error now contains gradient w.r.t embeddings)
    embedding_backward_kernel<<<num_blocks, block_size>>>(
        lm->d_embed_grad, ssm->d_error, d_input_batch,
        SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM
    );
    
    // Update weights
    // Update SSM
    update_weights_ssm(ssm, learning_rate);
    
    // Update embeddings and output weights
    num_blocks = (VOCAB_SIZE * EMBEDDING_DIM + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        lm->d_embeddings, lm->d_embed_grad, lm->d_embed_m, lm->d_embed_v,
        learning_rate, 0.9f, 0.999f, 1e-8f, weight_decay, adam_t, 
        VOCAB_SIZE * EMBEDDING_DIM, BATCH_SIZE
    );
    
    num_blocks = (EMBEDDING_DIM * VOCAB_SIZE + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        lm->d_output_weights, lm->d_output_grad, lm->d_output_m, lm->d_output_v,
        learning_rate, 0.9f, 0.999f, 1e-8f, weight_decay, adam_t,
        EMBEDDING_DIM * VOCAB_SIZE, BATCH_SIZE
    );
    
    // Cleanup
    cudaFree(d_embedded);
    cudaFree(d_logits);
    cudaFree(d_loss);
    cudaFree(d_logit_grad);
    
    return loss_val;
}

// Generate text
void generate_text(LanguageModel* lm, SSM* ssm, int start_token, int length) {
    // Allocate for single sequence generation
    int* d_input = NULL;
    float* d_embedded = NULL;
    float* d_logits = NULL;
    float* h_logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_embedded, EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
    
    // Initialize with start token
    CHECK_CUDA(cudaMemcpy(d_input, &start_token, sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Generated text: ");
    printf("%c", (char)start_token);
    
    // Temporary SSM for single-step generation
    SSM* gen_ssm = init_ssm(EMBEDDING_DIM, STATE_DIM, EMBEDDING_DIM, 1, 1);
    
    // Copy weights from trained model
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_A, ssm->d_A, STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_B, ssm->d_B, STATE_DIM * EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_C, ssm->d_C, EMBEDDING_DIM * STATE_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gen_ssm->d_D, ssm->d_D, EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    
    for (int i = 0; i < length - 1; i++) {
        // Embed
        embedding_forward_kernel<<<1, EMBEDDING_DIM>>>(
            d_embedded, lm->d_embeddings, d_input, 1, 1, EMBEDDING_DIM
        );
        
        // SSM forward
        forward_pass_ssm(gen_ssm, d_embedded);
        
        // Project to vocab
        CHECK_CUBLAS(cublasSgemv(lm->cublas_handle, CUBLAS_OP_T,
                                EMBEDDING_DIM, VOCAB_SIZE,
                                &(float){1.0f}, lm->d_output_weights, EMBEDDING_DIM,
                                gen_ssm->d_predictions, 1,
                                &(float){0.0f}, d_logits, 1));
        
        // Get logits and sample
        CHECK_CUDA(cudaMemcpy(h_logits, d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Simple argmax sampling
        int next_token = 0;
        float max_logit = h_logits[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (h_logits[j] > max_logit) {
                max_logit = h_logits[j];
                next_token = j;
            }
        }
        
        printf("%c", (char)next_token);
        CHECK_CUDA(cudaMemcpy(d_input, &next_token, sizeof(int), cudaMemcpyHostToDevice));
    }
    printf("\n");
    
    // Cleanup
    free(h_logits);
    cudaFree(d_input);
    cudaFree(d_embedded);
    cudaFree(d_logits);
    free_ssm(gen_ssm);
}

int main() {
    srand(time(NULL));
    
    // Load data
    int* d_all_inputs;
    int* d_all_targets;
    int num_batches;
    load_text_sequences("combined_corpus.txt", &d_all_inputs, &d_all_targets, 
                       &num_batches, SEQ_LEN, BATCH_SIZE);
    
    // Initialize model components
    SSM* ssm = init_ssm(EMBEDDING_DIM, STATE_DIM, EMBEDDING_DIM, SEQ_LEN, BATCH_SIZE);
    LanguageModel* lm = init_language_model(VOCAB_SIZE, EMBEDDING_DIM, ssm->cublas_handle);
    
    // Training
    const int num_epochs = 100;
    const float learning_rate = 0.001f;
    const float weight_decay = 0.01f;
    int adam_t = 0;
    
    printf("Starting training...\n");
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches && batch < 100; batch++) {  // Limit batches for testing
            adam_t++;
            
            // Get batch pointers
            int* d_input_batch = d_all_inputs + batch * BATCH_SIZE * SEQ_LEN;
            int* d_target_batch = d_all_targets + batch * BATCH_SIZE * SEQ_LEN;
            
            float loss = train_step(lm, ssm, d_input_batch, d_target_batch, 
                                  learning_rate, weight_decay, adam_t);
            epoch_loss += loss;
        }
        
        epoch_loss /= fminf(num_batches, 100);
        
        if (epoch % 10 == 0) {
            printf("Epoch %d/%d, Loss: %.4f, Perplexity: %.2f\n", 
                   epoch + 1, num_epochs, epoch_loss, expf(epoch_loss));
            
            // Generate sample
            generate_text(lm, ssm, 'T', 100);
        }
    }
    
    // Save model
    char timestamp[32];
    time_t now = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    char model_path[64];
    snprintf(model_path, sizeof(model_path), "slm_%s.bin", timestamp);
    save_ssm(ssm, model_path);
    
    // TODO: Save language model weights (embeddings, output weights)
    
    // Cleanup
    cudaFree(d_all_inputs);
    cudaFree(d_all_targets);
    free_language_model(lm);
    free_ssm(ssm);
    
    return 0;
}