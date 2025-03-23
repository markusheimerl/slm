#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

/* ------------------------- Data Structures ------------------------- */

typedef struct {
    // Token mixing parameters [seq_length x seq_length] (column–major)
    float* token_mixing_weight;
    float* token_mixing_weight_grad;
    float* token_mixing_m;
    float* token_mixing_v;
    
    // Channel mixing parameters [embed_dim x embed_dim] (column–major)
    float* channel_mixing_weight;
    float* channel_mixing_weight_grad;
    float* channel_mixing_m;
    float* channel_mixing_v;
    
    // Forward–pass buffers (many stored in row–major order)
    float* input_buffer;      // [batch, seq, embed] (row–major)
    float* residual;          // [batch, seq, embed]
    float* transposed;        // [batch, embed, seq] -> stored as: for each batch, element (e,s) is at index: s + (b*embed+e)*seq
    float* token_mixed;       // [batch, embed, seq]
    float* token_mix_activated; // [batch, embed, seq]
    float* channel_mixed;     // [batch, seq, embed]
    float* channel_mix_activated; // [batch, seq, embed]
    
    // Backward–pass buffers
    float* d_output;         // [batch, seq, embed]
    float* d_token_mixed;    // [batch, embed, seq]
    float* d_channel_mixed;  // [batch, seq, embed]
    float* d_input;          // [batch, seq, embed]
    
    // Additional buffers
    float* masked_weights;      // [seq, seq]
    float* d_channel_activated; // [batch, seq, embed]
    float* d_output_transposed; // [batch, embed, seq]
    float* d_input_transposed;  // [batch, embed, seq]
    float* temp_grad;           // [seq, seq]
     
    // Dimensions:
    int embed_dim;
    int seq_length;
} MixerBlock;

typedef struct {
    // Embedding parameters [vocab_size x embed_dim] (column–major)
    float* embedding_weight;
    float* embedding_weight_grad;
    float* embedding_m;
    float* embedding_v;
    
    // Output projection parameters [vocab_size x embed_dim] (column–major)
    float* out_proj_weight;
    float* out_proj_weight_grad;
    float* out_proj_m;
    float* out_proj_v;
    
    // Array of MixerBlocks
    MixerBlock** blocks;
    
    // Forward–pass buffers
    float* embeddings;     // [batch, seq, embed] (row–major)
    float* block_outputs;  // (num_layers+1) concatenated outputs, each [batch, seq, embed]
    float* logits;         // [batch, seq, vocab_size] (the GEMM uses column–major ordering)
    
    // Backward–pass buffers
    float* d_logits;       // [batch, seq, vocab_size]
    float* d_block_outputs;// same layout as block_outputs
    
    // Persistent buffers for training (CPU arrays)
    int* d_input_tokens;
    int* d_target_tokens;
    float* d_loss_buffer;
    
    // Host buffer for loss computation
    float* h_loss_buffer;
    
    // Adam optimizer parameters
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int t;  // time step counter
    
    // Dimensions/hyperparameters
    int vocab_size;
    int embed_dim;
    int num_layers;
    int seq_length;
    int batch_size;
} MixerModel;

/* ------------------------- CPU Helper Functions ------------------------- */

// Elementwise SiLU activation: out[i] = x * sigmoid(x)
void silu_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        output[i] = x * sig;
    }
}

// Multiply gradient by SiLU derivative: grad_out[i] = grad_in[i] * (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)))
void silu_deriv_mult_cpu(const float* pre, const float* grad_in, float* grad_out, int N) {
    for (int i = 0; i < N; i++) {
        float x = pre[i];
        float sig = 1.0f / (1.0f + expf(-x));
        float deriv = sig + x * sig * (1.0f - sig);
        grad_out[i] = grad_in[i] * deriv;
    }
}

// Transpose from [batch, seq, embed] (row–major) to [batch, embed, seq]
// For each batch b: for each sequence index s and embedding index e, 
// output index = b*(embed*seq) + e*seq + s.
void transpose_BSE_to_BES_cpu(const float* input, float* output, int batch, int seq, int embed) {
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int e = 0; e < embed; e++) {
                output[b * embed * seq + e * seq + s] = input[b * seq * embed + s * embed + e];
            }
        }
    }
}

// Transpose from [batch, embed, seq] to [batch, seq, embed]
void transpose_BES_to_BSE_cpu(const float* input, float* output, int batch, int embed, int seq) {
    for (int b = 0; b < batch; b++) {
        for (int e = 0; e < embed; e++) {
            for (int s = 0; s < seq; s++) {
                output[b * seq * embed + s * embed + e] = input[b * embed * seq + e * seq + s];
            }
        }
    }
}

// Apply lower–triangular mask: for a matrix in column–major order, set matrix[row,col]=0 if col > row.
void apply_lower_triangular_mask_cpu(float* matrix, int seq_length) {
    for (int row = 0; row < seq_length; row++) {
        for (int col = 0; col < seq_length; col++) {
            if (col > row) {
                matrix[row + col * seq_length] = 0.0f;
            }
        }
    }
}

// Embedding lookup: for each token, copy its row from the embedding_weight (column–major) into embeddings.
// For a given token, element at row=token and column e is stored at token + e * vocab_size.
void apply_embedding_cpu(const int* input_tokens, const float* embedding_weight, float* embeddings,
                         int batch_size, int seq_length, int embed_dim, int vocab_size) {
    for (int idx = 0; idx < batch_size * seq_length; idx++) {
        int token = input_tokens[idx];
        if (token < 0 || token >= vocab_size)
            token = 0;
        for (int e = 0; e < embed_dim; e++) {
            embeddings[idx * embed_dim + e] = embedding_weight[token + e * vocab_size];
        }
    }
}

// AdamW weight update (CPU loop).
void adamw_update_cpu(float* weight, const float* grad, float* m, float* v,
                     float beta1, float beta2, float epsilon, float learning_rate,
                     float weight_decay, float alpha_t, int size, float scale) {
    for (int i = 0; i < size; i++) {
        float g = grad[i] / scale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float update = alpha_t * m[i] / (sqrtf(v[i]) + epsilon);
        weight[i] = weight[i] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Compute softmax and cross-entropy loss and gradients.
void softmax_cross_entropy_cpu(const float* logits, const int* targets,
                              float* d_logits, float* loss_buffer,
                              int batch_size, int seq_length, int vocab_size) {
    int total = batch_size * seq_length;
    for (int i = 0; i < total; i++) {
        int target = targets[i];
        const float* logits_row = logits + i * vocab_size;
        float* d_logits_row = d_logits + i * vocab_size;
        float max_val = logits_row[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits_row[v] > max_val) max_val = logits_row[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float exp_val = expf(logits_row[v] - max_val);
            d_logits_row[v] = exp_val; // store temporarily
            sum_exp += exp_val;
        }
        float loss = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float prob = d_logits_row[v] / sum_exp;
            d_logits_row[v] = prob;
            if (v == target) {
                d_logits_row[v] -= 1.0f;
                loss = -logf(prob + 1e-10f);
            }
        }
        loss_buffer[i] = loss;
    }
}

// Embedding backward: for each token in input_tokens, add gradient from d_embeddings.
void embedding_backward_cpu(const int* input_tokens, const float* d_embeddings,
                           float* embedding_grad, int batch_size, int seq_length,
                           int embed_dim, int vocab_size) {
    for (int idx = 0; idx < batch_size * seq_length; idx++) {
        int token = input_tokens[idx];
        if (token >= 0 && token < vocab_size) {
            for (int e = 0; e < embed_dim; e++) {
                embedding_grad[token + e * vocab_size] += d_embeddings[idx * embed_dim + e];
            }
        }
    }
}

/* ------------------------- Initialization and Free ------------------------- */

MixerBlock* init_mixer_block(int embed_dim, int seq_length, int batch_size) {
    MixerBlock* block = (MixerBlock*)malloc(sizeof(MixerBlock));
    block->embed_dim = embed_dim;
    block->seq_length = seq_length;
    
    size_t size_tok = seq_length * seq_length * sizeof(float);
    float scale_token = 1.0f / sqrtf((float)seq_length);
    float* h_token_mixing_weight = (float*)malloc(size_tok);
    for (int i = 0; i < seq_length * seq_length; i++)
        h_token_mixing_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_token);
    block->token_mixing_weight = (float*)malloc(size_tok);
    memcpy(block->token_mixing_weight, h_token_mixing_weight, size_tok);
    free(h_token_mixing_weight);
    
    block->token_mixing_weight_grad = (float*)malloc(size_tok);
    memset(block->token_mixing_weight_grad, 0, size_tok);
    block->token_mixing_m = (float*)malloc(size_tok);
    memset(block->token_mixing_m, 0, size_tok);
    block->token_mixing_v = (float*)malloc(size_tok);
    memset(block->token_mixing_v, 0, size_tok);
    
    block->masked_weights = (float*)malloc(size_tok);
    
    size_t size_channel = embed_dim * embed_dim * sizeof(float);
    float scale_channel = 1.0f / sqrtf((float)embed_dim);
    float* h_channel_mixing_weight = (float*)malloc(size_channel);
    for (int i = 0; i < embed_dim * embed_dim; i++)
        h_channel_mixing_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_channel);
    block->channel_mixing_weight = (float*)malloc(size_channel);
    memcpy(block->channel_mixing_weight, h_channel_mixing_weight, size_channel);
    free(h_channel_mixing_weight);
    
    block->channel_mixing_weight_grad = (float*)malloc(size_channel);
    memset(block->channel_mixing_weight_grad, 0, size_channel);
    block->channel_mixing_m = (float*)malloc(size_channel);
    memset(block->channel_mixing_m, 0, size_channel);
    block->channel_mixing_v = (float*)malloc(size_channel);
    memset(block->channel_mixing_v, 0, size_channel);
    
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    block->input_buffer = (float*)malloc(tensor_size);
    block->residual = (float*)malloc(tensor_size);
    
    size_t tensor_size_trans = batch_size * embed_dim * seq_length * sizeof(float);
    block->transposed = (float*)malloc(tensor_size_trans);
    block->token_mixed = (float*)malloc(tensor_size_trans);
    block->token_mix_activated = (float*)malloc(tensor_size_trans);
    block->channel_mixed = (float*)malloc(tensor_size);
    block->channel_mix_activated = (float*)malloc(tensor_size);
    
    block->d_output = (float*)malloc(tensor_size);
    block->d_token_mixed = (float*)malloc(tensor_size_trans);
    block->d_channel_mixed = (float*)malloc(tensor_size);
    block->d_input = (float*)malloc(tensor_size);
    
    block->d_channel_activated = (float*)malloc(tensor_size);
    block->d_output_transposed = (float*)malloc(tensor_size_trans);
    block->d_input_transposed = (float*)malloc(tensor_size_trans);
    block->temp_grad = (float*)malloc(size_tok);
    memset(block->temp_grad, 0, size_tok);
    
    return block;
}

void free_mixer_block(MixerBlock* block) {
    free(block->token_mixing_weight);
    free(block->token_mixing_weight_grad);
    free(block->token_mixing_m);
    free(block->token_mixing_v);
    
    free(block->channel_mixing_weight);
    free(block->channel_mixing_weight_grad);
    free(block->channel_mixing_m);
    free(block->channel_mixing_v);
    
    free(block->input_buffer);
    free(block->residual);
    free(block->transposed);
    free(block->token_mixed);
    free(block->token_mix_activated);
    free(block->channel_mixed);
    free(block->channel_mix_activated);
    
    free(block->d_output);
    free(block->d_token_mixed);
    free(block->d_channel_mixed);
    free(block->d_input);
    
    free(block->masked_weights);
    free(block->d_channel_activated);
    free(block->d_output_transposed);
    free(block->d_input_transposed);
    free(block->temp_grad);
    
    free(block);
}

MixerModel* init_mixer_model(int vocab_size, int embed_dim, int num_layers, int seq_length, int batch_size) {
    MixerModel* model = (MixerModel*)malloc(sizeof(MixerModel));
    model->vocab_size = vocab_size;
    model->embed_dim = embed_dim;
    model->num_layers = num_layers;
    model->seq_length = seq_length;
    model->batch_size = batch_size;
    
    model->beta1 = 0.9f;
    model->beta2 = 0.999f;
    model->epsilon = 1e-8f;
    model->weight_decay = 0.01f;
    model->t = 0;
    
    size_t embed_matrix_size = vocab_size * embed_dim * sizeof(float);
    float scale_embed = 1.0f / sqrtf((float)embed_dim);
    float* h_embedding_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++)
        h_embedding_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    model->embedding_weight = (float*)malloc(embed_matrix_size);
    memcpy(model->embedding_weight, h_embedding_weight, embed_matrix_size);
    free(h_embedding_weight);
    
    model->embedding_weight_grad = (float*)malloc(embed_matrix_size);
    memset(model->embedding_weight_grad, 0, embed_matrix_size);
    model->embedding_m = (float*)malloc(embed_matrix_size);
    memset(model->embedding_m, 0, embed_matrix_size);
    model->embedding_v = (float*)malloc(embed_matrix_size);
    memset(model->embedding_v, 0, embed_matrix_size);
    
    float* h_out_proj_weight = (float*)malloc(embed_matrix_size);
    for (int i = 0; i < vocab_size * embed_dim; i++)
        h_out_proj_weight[i] = (((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale_embed);
    model->out_proj_weight = (float*)malloc(embed_matrix_size);
    memcpy(model->out_proj_weight, h_out_proj_weight, embed_matrix_size);
    free(h_out_proj_weight);
    
    model->out_proj_weight_grad = (float*)malloc(embed_matrix_size);
    memset(model->out_proj_weight_grad, 0, embed_matrix_size);
    model->out_proj_m = (float*)malloc(embed_matrix_size);
    memset(model->out_proj_m, 0, embed_matrix_size);
    model->out_proj_v = (float*)malloc(embed_matrix_size);
    memset(model->out_proj_v, 0, embed_matrix_size);
    
    model->blocks = (MixerBlock**)malloc(num_layers * sizeof(MixerBlock*));
    for (int i = 0; i < num_layers; i++)
        model->blocks[i] = init_mixer_block(embed_dim, seq_length, batch_size);
    
    size_t tensor_size = batch_size * seq_length * embed_dim * sizeof(float);
    model->embeddings = (float*)malloc(tensor_size);
    model->block_outputs = (float*)malloc((num_layers+1) * tensor_size);
    
    size_t logits_size = batch_size * seq_length * vocab_size * sizeof(float);
    model->logits = (float*)malloc(logits_size);
    
    model->d_logits = (float*)malloc(logits_size);
    model->d_block_outputs = (float*)malloc((num_layers+1) * tensor_size);
    
    model->d_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    model->d_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    model->d_loss_buffer = (float*)malloc(batch_size * seq_length * sizeof(float));
    
    model->h_loss_buffer = (float*)malloc(batch_size * seq_length * sizeof(float));
    
    return model;
}

void free_mixer_model(MixerModel* model) {
    free(model->embedding_weight);
    free(model->embedding_weight_grad);
    free(model->embedding_m);
    free(model->embedding_v);
    
    free(model->out_proj_weight);
    free(model->out_proj_weight_grad);
    free(model->out_proj_m);
    free(model->out_proj_v);
    
    for (int i = 0; i < model->num_layers; i++) {
        free_mixer_block(model->blocks[i]);
    }
    free(model->blocks);
    
    free(model->embeddings);
    free(model->block_outputs);
    free(model->logits);
    
    free(model->d_logits);
    free(model->d_block_outputs);
    
    free(model->d_input_tokens);
    free(model->d_target_tokens);
    free(model->d_loss_buffer);
    free(model->h_loss_buffer);
    
    free(model);
}

/* ------------------------- Forward and Backward Pass ------------------------- */

// Forward pass through one MixerBlock.
void mixer_block_forward(MixerBlock* block, float* input, float* output, int batch_size) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int total = batch_size * seq * embed;
    int total_trans = batch_size * embed * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    memcpy(block->input_buffer, input, total * sizeof(float));
    memcpy(block->residual, input, total * sizeof(float));
    
    transpose_BSE_to_BES_cpu(input, block->transposed, batch_size, seq, embed);
    
    memcpy(block->masked_weights, block->token_mixing_weight, seq * seq * sizeof(float));
    apply_lower_triangular_mask_cpu(block->masked_weights, seq);
    
    int combined_batch_embed = batch_size * embed;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                seq, combined_batch_embed, seq,
                alpha,
                block->masked_weights, seq,
                block->transposed, seq,
                beta,
                block->token_mixed, seq);
    
    silu_cpu(block->token_mixed, block->token_mix_activated, total_trans);
    
    transpose_BES_to_BSE_cpu(block->token_mix_activated, output, batch_size, embed, seq);
    
    cblas_saxpy(total, alpha, block->residual, 1, output, 1);
    
    // --- Channel Mixing ---
    memcpy(block->residual, output, total * sizeof(float));
    
    int combined_batch = batch_size * seq;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                embed, combined_batch, embed,
                alpha,
                block->channel_mixing_weight, embed,
                output, embed,
                beta,
                block->channel_mixed, embed);
    
    silu_cpu(block->channel_mixed, block->channel_mix_activated, total);
    
    memcpy(output, block->channel_mix_activated, total * sizeof(float));
    
    cblas_saxpy(total, alpha, block->residual, 1, output, 1);
}

// Forward pass through the entire model.
void mixer_model_forward(MixerModel* model, int* input_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int total = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    apply_embedding_cpu(input_tokens, model->embedding_weight, model->embeddings,
                         batch, seq, embed, model->vocab_size);
    
    int tensor_elements = total * embed;
    memcpy(model->block_outputs, model->embeddings, tensor_elements * sizeof(float));
    
    for (int i = 0; i < model->num_layers; i++) {
        float* input_ptr = model->block_outputs + i * tensor_elements;
        float* output_ptr = model->block_outputs + (i + 1) * tensor_elements;
        mixer_block_forward(model->blocks[i], input_ptr, output_ptr, batch);
    }
    
    // Output projection: logits = final_output * (out_proj_weight)^T
    float* final_output = model->block_outputs + model->num_layers * tensor_elements;
    int combined_batch_seq = batch * seq;
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                model->vocab_size, combined_batch_seq, embed,
                alpha,
                model->out_proj_weight, embed,
                final_output, embed,
                beta,
                model->logits, model->vocab_size);
}

// Compute cross–entropy loss and gradients.
float compute_loss_and_gradients(MixerModel* model, int* target_tokens) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int vocab = model->vocab_size;
    int total = batch * seq;
    
    softmax_cross_entropy_cpu(model->logits, target_tokens,
                              model->d_logits, model->d_loss_buffer,
                              batch, seq, vocab);
    
    memcpy(model->h_loss_buffer, model->d_loss_buffer, total * sizeof(float));
    float total_loss = 0.0f;
    for (int i = 0; i < total; i++) {
        total_loss += model->h_loss_buffer[i];
    }
    return total_loss / total;
}

// Backward pass through one MixerBlock.
void mixer_block_backward(MixerBlock* block, float* d_output, float* d_input, int batch_size) {
    int seq = block->seq_length;
    int embed = block->embed_dim;
    int total = batch_size * seq * embed;
    int total_trans = batch_size * embed * seq;
    int combined_batch = batch_size * seq;
    int combined_batch_embed = batch_size * embed;
    float alpha = 1.0f, beta = 0.0f;
    
    memcpy(block->d_output, d_output, total * sizeof(float));
    memcpy(block->d_channel_activated, block->d_output, total * sizeof(float));
    silu_deriv_mult_cpu(block->channel_mixed, block->d_channel_activated, block->d_channel_mixed, total);
    
    // Gradient w.r.t. channel mixing weights:
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                embed, embed, combined_batch,
                alpha,
                block->d_channel_mixed, embed,
                block->residual, embed,
                beta,
                block->channel_mixing_weight_grad, embed);
    
    // Gradient w.r.t. inputs from channel mixing:
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                embed, combined_batch, embed,
                alpha,
                block->channel_mixing_weight, embed,
                block->d_channel_mixed, embed,
                beta,
                d_input, embed);
    
    cblas_saxpy(total, alpha, block->d_output, 1, d_input, 1);
    
    memcpy(block->d_output, d_input, total * sizeof(float));
    transpose_BSE_to_BES_cpu(d_input, block->d_output_transposed, batch_size, seq, embed);
    silu_deriv_mult_cpu(block->token_mixed, block->d_output_transposed, block->d_token_mixed, total_trans);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                seq, seq, combined_batch_embed,
                alpha,
                block->transposed, seq,
                block->d_token_mixed, seq,
                beta,
                block->temp_grad, seq);
    
    apply_lower_triangular_mask_cpu(block->temp_grad, seq);
    
    cblas_saxpy(seq * seq, alpha, block->temp_grad, 1, block->token_mixing_weight_grad, 1);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                seq, combined_batch_embed, seq,
                alpha,
                block->masked_weights, seq,
                block->d_token_mixed, seq,
                beta,
                block->d_input_transposed, seq);
    
    transpose_BES_to_BSE_cpu(block->d_input_transposed, d_input, batch_size, embed, seq);
    
    cblas_saxpy(total, alpha, block->d_output, 1, d_input, 1);
}

// Backward pass through the entire model.
void mixer_model_backward(MixerModel* model) {
    int batch = model->batch_size;
    int seq = model->seq_length;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    int total = batch * seq * embed;
    int combined_batch_seq = batch * seq;
    float alpha = 1.0f, beta = 0.0f;
    
    float* final_output = model->block_outputs + model->num_layers * total;
    float* d_final_output = model->d_block_outputs + model->num_layers * total;
    memset(d_final_output, 0, (combined_batch_seq * embed) * sizeof(float));
    
    // Gradient w.r.t. output projection weights
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                embed, vocab, combined_batch_seq,
                alpha,
                final_output, embed,
                model->d_logits, vocab,
                beta,
                model->out_proj_weight_grad, embed);
    
    // Gradient w.r.t. final output
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                embed, combined_batch_seq, vocab,
                alpha,
                model->out_proj_weight, embed,
                model->d_logits, vocab,
                beta,
                d_final_output, embed);
    
    for (int i = model->num_layers - 1; i >= 0; i--) {
        float* d_output = model->d_block_outputs + (i + 1) * total;
        float* d_input = model->d_block_outputs + i * total;
        mixer_block_backward(model->blocks[i], d_output, d_input, batch);
    }
    
    memset(model->embedding_weight_grad, 0, model->vocab_size * embed * sizeof(float));
    embedding_backward_cpu(model->d_input_tokens, model->d_block_outputs,
                           model->embedding_weight_grad, batch, seq, embed, model->vocab_size);
}

/* ------------------------- Weight Update and Gradient Zero ------------------------- */

void update_weights_adamw(MixerModel* model, float learning_rate) {
    model->t++;
    float beta1_t = powf(model->beta1, model->t);
    float beta2_t = powf(model->beta2, model->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    float scale = model->batch_size * model->seq_length;
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    
    adamw_update_cpu(model->embedding_weight, model->embedding_weight_grad,
                    model->embedding_m, model->embedding_v,
                    model->beta1, model->beta2, model->epsilon,
                    learning_rate, model->weight_decay, alpha_t,
                    vocab * embed, scale);
    
    adamw_update_cpu(model->out_proj_weight, model->out_proj_weight_grad,
                    model->out_proj_m, model->out_proj_v,
                    model->beta1, model->beta2, model->epsilon,
                    learning_rate, model->weight_decay, alpha_t,
                    vocab * embed, scale);
    
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        int size_tok = block->seq_length * block->seq_length;
        int size_channel = block->embed_dim * block->embed_dim;
        
        adamw_update_cpu(block->token_mixing_weight, block->token_mixing_weight_grad,
                        block->token_mixing_m, block->token_mixing_v,
                        model->beta1, model->beta2, model->epsilon,
                        learning_rate, model->weight_decay, alpha_t,
                        size_tok, scale);
        
        adamw_update_cpu(block->channel_mixing_weight, block->channel_mixing_weight_grad,
                        block->channel_mixing_m, block->channel_mixing_v,
                        model->beta1, model->beta2, model->epsilon,
                        learning_rate, model->weight_decay, alpha_t,
                        size_channel, scale);
    }
}

void zero_gradients(MixerModel* model) {
    int embed = model->embed_dim;
    int vocab = model->vocab_size;
    size_t size = vocab * embed * sizeof(float);
    memset(model->embedding_weight_grad, 0, size);
    memset(model->out_proj_weight_grad, 0, size);
    for (int l = 0; l < model->num_layers; l++) {
        MixerBlock* block = model->blocks[l];
        size_t size_tok = block->seq_length * block->seq_length * sizeof(float);
        size_t size_chan = block->embed_dim * block->embed_dim * sizeof(float);
        memset(block->token_mixing_weight_grad, 0, size_tok);
        memset(block->channel_mixing_weight_grad, 0, size_chan);
    }
}

/* ------------------------- Data Loading and Sampling ------------------------- */

// Load text data from a file.
char* load_text_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error: could not read entire file\n");
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[file_size] = '\0';
    *size = file_size;
    fclose(file);
    return buffer;
}

// Get a batch of randomly sampled sequences from the text.
void get_random_batch(const char* text, size_t text_size, int batch_size, int seq_length, 
                      int* input_tokens, int* target_tokens) {
    if (text_size <= (size_t)seq_length) {
        fprintf(stderr, "Error: Text size is too small for the sequence length\n");
        return;
    }
    
    for (int b = 0; b < batch_size; b++) {
        int start_pos = rand() % (text_size - seq_length - 1);
        for (int s = 0; s < seq_length; s++) {
            input_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s];
            target_tokens[b * seq_length + s] = (unsigned char)text[start_pos + s + 1];
        }
    }
}

int count_parameters(MixerModel* model) {
    int total_params = 0;
    total_params += model->vocab_size * model->embed_dim;
    total_params += model->vocab_size * model->embed_dim;
    for (int i = 0; i < model->num_layers; i++) {
        total_params += model->seq_length * model->seq_length;
        total_params += model->embed_dim * model->embed_dim;
    }
    return total_params;
}

/* ------------------------- Model Save/Load ------------------------- */

void save_model(MixerModel* model, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for writing\n", filename);
        return;
    }
    
    fwrite(&model->vocab_size, sizeof(int), 1, file);
    fwrite(&model->embed_dim, sizeof(int), 1, file);
    fwrite(&model->num_layers, sizeof(int), 1, file);
    fwrite(&model->seq_length, sizeof(int), 1, file);
    fwrite(&model->batch_size, sizeof(int), 1, file);
    
    int vocab = model->vocab_size, embed = model->embed_dim;
    fwrite(model->embedding_weight, sizeof(float), vocab * embed, file);
    fwrite(model->out_proj_weight, sizeof(float), vocab * embed, file);
    
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int seq = block->seq_length;
        int size_tok = seq * seq;
        fwrite(block->token_mixing_weight, sizeof(float), size_tok, file);
        int size_channel = embed * embed;
        fwrite(block->channel_mixing_weight, sizeof(float), size_channel, file);
    }
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

MixerModel* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: could not open file %s for reading\n", filename);
        return NULL;
    }
    
    int vocab_size, embed_dim, num_layers, seq_length, batch_size;
    fread(&vocab_size, sizeof(int), 1, file);
    fread(&embed_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&seq_length, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    
    fread(model->embedding_weight, sizeof(float), vocab_size * embed_dim, file);
    fread(model->out_proj_weight, sizeof(float), vocab_size * embed_dim, file);
    
    for (int i = 0; i < model->num_layers; i++) {
        MixerBlock* block = model->blocks[i];
        int size_tok = seq_length * seq_length;
        fread(block->token_mixing_weight, sizeof(float), size_tok, file);
        int size_channel = embed_dim * embed_dim;
        fread(block->channel_mixing_weight, sizeof(float), size_channel, file);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return model;
}

/* ------------------------- Text Generation and LR Schedule ------------------------- */

void generate_text(MixerModel* model, const char* corpus, size_t corpus_size, int max_new_tokens, float temperature) {
    int seq_length = model->seq_length;
    int* h_tokens = (int*)malloc(seq_length * sizeof(int));
    
    int start_pos = rand() % (corpus_size - seq_length);
    
    for (int i = 0; i < seq_length; i++) {
        h_tokens[i] = (unsigned char)corpus[start_pos + i];
        printf("%c", h_tokens[i]);
    }
    
    memcpy(model->d_input_tokens, h_tokens, seq_length * sizeof(int));
    
    for (int i = 0; i < max_new_tokens; i++) {
        mixer_model_forward(model, model->d_input_tokens);
        
        float* h_logits = (float*)malloc(model->vocab_size * sizeof(float));
        memcpy(h_logits, model->logits + (seq_length - 1) * model->vocab_size,
               model->vocab_size * sizeof(float));
        
        float temp = temperature + 1e-7f;
        for (int v = 0; v < model->vocab_size; v++)
            h_logits[v] /= temp;
        
        float max_logit = h_logits[0];
        for (int v = 1; v < model->vocab_size; v++) {
            if (h_logits[v] > max_logit)
                max_logit = h_logits[v];
        }
        
        float sum_exp = 0.0f;
        for (int v = 0; v < model->vocab_size; v++) {
            h_logits[v] = expf(h_logits[v] - max_logit);
            sum_exp += h_logits[v];
        }
        for (int v = 0; v < model->vocab_size; v++)
            h_logits[v] /= sum_exp;
        
        float r = (float)rand() / (float)RAND_MAX;
        float cdf = 0.0f;
        int next_token = 0;
        for (int v = 0; v < model->vocab_size; v++) {
            cdf += h_logits[v];
            if (r < cdf) {
                next_token = v;
                break;
            }
        }
        
        printf("%c", next_token);
        fflush(stdout);
        
        for (int j = 0; j < seq_length - 1; j++)
            h_tokens[j] = h_tokens[j + 1];
        h_tokens[seq_length - 1] = next_token;
        
        memcpy(model->d_input_tokens, h_tokens, seq_length * sizeof(int));
        free(h_logits);
    }
    
    printf("\n");
    free(h_tokens);
}

float get_cosine_lr(float initial_lr, float min_lr, int current_step, int total_steps) {
    float progress = (float)current_step / (float)total_steps;
    return min_lr + 0.5f * (initial_lr - min_lr) * (1.0f + cosf(M_PI * progress));
}

/* ------------------------- Main Function ------------------------- */

int main() {
    srand(time(NULL));
    
    int vocab_size = 256;
    int embed_dim = 128;
    int num_layers = 2;
    int seq_length = 256;
    int batch_size = 4;
    
    MixerModel* model = init_mixer_model(vocab_size, embed_dim, num_layers, seq_length, batch_size);
    printf("Model initialized with %d parameters\n", count_parameters(model));
    
    size_t text_size;
    char* text = load_text_file("gutenberg_texts/combined_corpus.txt", &text_size);
    printf("Loaded text corpus with %zu bytes\n", text_size);
    
    float initial_lr = 0.0001f;
    float min_lr = 0.00001f;
    int total_training_steps = 10000;
    
    printf("Training for %d total steps with cosine learning rate schedule (%.6f to %.6f)\n", 
           total_training_steps, initial_lr, min_lr);
    
    int* h_input_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    int* h_target_tokens = (int*)malloc(batch_size * seq_length * sizeof(int));
    
    time_t start_time = time(NULL);
    
    for (int step = 0; step < total_training_steps; step++) {
        float learning_rate = get_cosine_lr(initial_lr, min_lr, step, total_training_steps);
        
        get_random_batch(text, text_size, batch_size, seq_length, h_input_tokens, h_target_tokens);
        
        memcpy(model->d_input_tokens, h_input_tokens, batch_size * seq_length * sizeof(int));
        memcpy(model->d_target_tokens, h_target_tokens, batch_size * seq_length * sizeof(int));
        
        mixer_model_forward(model, model->d_input_tokens);
        
        float step_loss = compute_loss_and_gradients(model, model->d_target_tokens);
        
        mixer_model_backward(model);
        update_weights_adamw(model, learning_rate);
        zero_gradients(model);
        
        if (step % 10 == 0)
            printf("Step %d/%d, LR: %.6f, Loss: %.4f\n", step, total_training_steps, learning_rate, step_loss);
        
        if (step % 200 == 0 && step > 0) {
            time_t current_time = time(NULL);
            printf("\n======= Sample generation at step %d after %ld seconds =======\n", step, current_time - start_time);
            generate_text(model, text, text_size, 128, 0.8f);
            printf("\n");
        }
    }
    
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_mixer_model.bin", localtime(&now));
    save_model(model, model_fname);
    
    free(h_input_tokens);
    free(h_target_tokens);
    free(text);
    free_mixer_model(model);
    
    return 0;
}