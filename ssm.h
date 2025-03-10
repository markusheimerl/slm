#ifndef SSM_H
#define SSM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

// ---------------------------------------------------------------------
// Structure definition for the state-space model (SSM)
// This version uses CBLAS and integrates the AdamW optimizer.
// Internally parameterizes A for stability.
// ---------------------------------------------------------------------
typedef struct {
    // State transition matrices
    float* A;          // state_dim x state_dim
    float* B;          // state_dim x input_dim
    float* C;          // output_dim x state_dim
    float* D;          // output_dim x input_dim

    // RMS Norm parameters
    float* rms_weight;    // state_dim
    float* rms_weight_grad; // state_dim
    float* rms_weight_m;  // Adam state for RMS weight
    float* rms_weight_v;  // Adam state for RMS weight
    float rms_eps;        // epsilon for numerical stability

    // Gradients
    float* A_grad;
    float* B_grad;
    float* C_grad;
    float* D_grad;

    // Adam optimizer first (m) and second (v) moment estimates
    float* A_m;
    float* A_v;
    float* B_m;
    float* B_v;
    float* C_m;
    float* C_v;
    float* D_m;
    float* D_v;

    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter

    // Helper arrays
    float* state;         // batch_size x state_dim
    float* next_state;    // batch_size x state_dim
    float* pre_state;     // batch_size x state_dim (pre-activation state)
    float* norm_state;    // batch_size x state_dim (normalized state)
    float* predictions;   // batch_size x output_dim
    float* error;         // batch_size x output_dim
    float* state_error;   // batch_size x state_dim
    float* rms_denom;     // batch_size (denominator for RMS norm)
    float* rms_grad;      // batch_size x state_dim (gradient of RMS norm)

    // Temporary buffers for matrix operations
    float* temp_state;    // batch_size x state_dim
    float* temp_output;   // batch_size x output_dim
    float* A_stable;      // Internal stable version of A

    // Dimensions of the network
    int input_dim;
    int state_dim;
    int output_dim;
    int batch_size;
} SSM;

// ---------------------------------------------------------------------
// Function: compute_stable_A
// Compute stable A matrix using tanh-based parameterization
// ---------------------------------------------------------------------
void compute_stable_A(float* A_stable, const float* A, int n) {
    for (int idx = 0; idx < n*n; idx++) {
        int row = idx / n;
        int col = idx % n;
        
        if (row == col) {
            // Diagonal elements: scaled tanh for eigenvalue control
            A_stable[idx] = 0.9f * tanhf(A[idx]);
        } else {
            // Off-diagonal elements: scaled by matrix size
            A_stable[idx] = A[idx] / sqrtf((float)n);
        }
    }
}

// ---------------------------------------------------------------------
// Function: compute_A_grad_from_stable_grad
// Compute gradient for A analytically
// ---------------------------------------------------------------------
void compute_A_grad_from_stable_grad(float* A_grad, 
                                    const float* A_stable_grad, 
                                    const float* A, int n) {
    for (int idx = 0; idx < n*n; idx++) {
        int row = idx / n;
        int col = idx % n;
        
        if (row == col) {
            // Diagonal derivative: d(tanh)/dA = sech²(A)
            float tanh_val = tanhf(A[idx]);
            float sech_squared = 1.0f - tanh_val * tanh_val;
            A_grad[idx] = A_stable_grad[idx] * 0.9f * sech_squared;
        } else {
            // Off-diagonal derivative: 1/sqrt(n)
            A_grad[idx] = A_stable_grad[idx] / sqrtf((float)n);
        }
    }
}

// ---------------------------------------------------------------------
// Function: Calculate squared norms per row (for RMS normalization)
// ---------------------------------------------------------------------
void rms_norm_squared(float* rms_denom, const float* pre_state, 
                      int batch_size, int state_dim) {
    for (int row = 0; row < batch_size; row++) {
        float squared_sum = 0.0f;
        for (int col = 0; col < state_dim; col++) {
            float val = pre_state[row * state_dim + col]; // Row-major
            squared_sum += val * val;
        }
        rms_denom[row] = squared_sum / state_dim;
    }
}

// ---------------------------------------------------------------------
// Function: Apply RMS normalization
// ---------------------------------------------------------------------
void rms_norm(float* norm_state, const float* pre_state, 
              const float* rms_denom, const float* rms_weight,
              float eps, int batch_size, int state_dim) {
    for (int row = 0; row < batch_size; row++) {
        float denom = sqrtf(rms_denom[row] + eps);
        for (int col = 0; col < state_dim; col++) {
            float val = pre_state[row * state_dim + col]; // Row-major
            norm_state[row * state_dim + col] = val * rms_weight[col] / denom;
        }
    }
}

// ---------------------------------------------------------------------
// Function: Calculate gradients for RMS normalization
// ---------------------------------------------------------------------
void rms_norm_backward(float* pre_state_grad, float* rms_weight_grad,
                      const float* state_error, const float* pre_state,
                      const float* rms_denom, const float* rms_weight,
                      float eps, int batch_size, int state_dim) {
    // Clear rms_weight_grad
    memset(rms_weight_grad, 0, state_dim * sizeof(float));
    
    for (int row = 0; row < batch_size; row++) {
        float denom = sqrtf(rms_denom[row] + eps);
        
        // First, calculate contribution to rms_weight gradients
        for (int col = 0; col < state_dim; col++) {
            float val = pre_state[row * state_dim + col];
            float grad = state_error[row * state_dim + col];
            rms_weight_grad[col] += grad * val / denom;
        }
        
        // Calculate factor2 (gradient contribution from the squared norm)
        float factor2_sum = 0.0f;
        for (int k = 0; k < state_dim; k++) {
            float grad_k = state_error[row * state_dim + k];
            float val_k = pre_state[row * state_dim + k];
            factor2_sum += grad_k * val_k * rms_weight[k];
        }
        
        // Calculate gradients for pre_state
        for (int col = 0; col < state_dim; col++) {
            float val = pre_state[row * state_dim + col];
            float grad = state_error[row * state_dim + col];
            
            float factor1 = rms_weight[col] / denom;
            float factor2 = -val * factor2_sum / (2.0f * state_dim * denom * denom * denom);
            
            pre_state_grad[row * state_dim + col] = grad * factor1 + factor2;
        }
    }
}

// ---------------------------------------------------------------------
// Function: swish_forward
// swish(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------
void swish_forward(float* output, const float* input, int size) {
    for (int idx = 0; idx < size; idx++) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

// ---------------------------------------------------------------------
// Function: swish_backward
// Computes derivative using: grad_output *= swish + sigmoid*(1-swish)
// ---------------------------------------------------------------------
void swish_backward(float* grad_output, const float* input, 
                   const float* activated, int size) {
    for (int idx = 0; idx < size; idx++) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float swish = activated[idx];  // already computed activated value
        grad_output[idx] *= (swish + sigmoid * (1.0f - swish));
    }
}

// ---------------------------------------------------------------------
// Function: mse_loss
// Mean Squared Error loss computation (elementwise error)
// ---------------------------------------------------------------------
void mse_loss(float* error, const float* predictions, 
             const float* targets, int size) {
    for (int idx = 0; idx < size; idx++) {
        float diff = predictions[idx] - targets[idx];
        error[idx] = diff;
    }
}

// ---------------------------------------------------------------------
// Function: adamw_update
// AdamW update (per weight element)
// ---------------------------------------------------------------------
void adamw_update(float* W, const float* grad, float* m, float* v, 
                 int size, float beta1, float beta2, float epsilon, 
                 float weight_decay, float learning_rate, int batch_size, 
                 float bias_correction1, float bias_correction2) {
    for (int idx = 0; idx < size; idx++) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// Function: init_ssm
// Initializes the SSM structure, allocates memory,
// sets initial weights with scaled random values.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
SSM* init_ssm(int input_dim, int state_dim, int output_dim, int batch_size) {
    SSM* ssm = (SSM*)malloc(sizeof(SSM));
    ssm->input_dim = input_dim;
    ssm->state_dim = state_dim;
    ssm->output_dim = output_dim;
    ssm->batch_size = batch_size;

    // Set Adam hyperparameters
    ssm->beta1 = 0.9f;
    ssm->beta2 = 0.999f;
    ssm->epsilon = 1e-8f;
    ssm->weight_decay = 0.01f;
    ssm->adam_t = 0;
    ssm->rms_eps = 1e-5f;

    // Allocate memory for weight matrices
    ssm->A = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->rms_weight = (float*)malloc(state_dim * sizeof(float));

    // Initialize matrices with scaled random values
    float scale_A = 1.0f / sqrtf(state_dim);
    float scale_B = 1.0f / sqrtf(input_dim);
    float scale_C = 1.0f / sqrtf(state_dim);
    float scale_D = 1.0f / sqrtf(input_dim);

    for (int i = 0; i < state_dim * state_dim; i++) {
        ssm->A[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_A;
    }
    for (int i = 0; i < state_dim * input_dim; i++) {
        ssm->B[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_B;
    }
    for (int i = 0; i < output_dim * state_dim; i++) {
        ssm->C[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_C;
    }
    for (int i = 0; i < output_dim * input_dim; i++) {
        ssm->D[i] = (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale_D;
    }
    // Initialize RMS Norm weights to 1.0
    for (int i = 0; i < state_dim; i++) {
        ssm->rms_weight[i] = 1.0f;
    }

    // Allocate memory for gradients
    ssm->A_grad = (float*)malloc(state_dim * state_dim * sizeof(float));
    ssm->B_grad = (float*)malloc(state_dim * input_dim * sizeof(float));
    ssm->C_grad = (float*)malloc(output_dim * state_dim * sizeof(float));
    ssm->D_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    ssm->rms_weight_grad = (float*)malloc(state_dim * sizeof(float));

    // Allocate memory for Adam first and second moment estimates and initialize to zero
    ssm->A_m = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->A_v = (float*)calloc(state_dim * state_dim, sizeof(float));
    ssm->B_m = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->B_v = (float*)calloc(state_dim * input_dim, sizeof(float));
    ssm->C_m = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->C_v = (float*)calloc(output_dim * state_dim, sizeof(float));
    ssm->D_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->D_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    ssm->rms_weight_m = (float*)calloc(state_dim, sizeof(float));
    ssm->rms_weight_v = (float*)calloc(state_dim, sizeof(float));

    // Allocate helper arrays
    ssm->state = (float*)calloc(batch_size * state_dim, sizeof(float));
    ssm->next_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->pre_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->norm_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->state_error = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->rms_denom = (float*)malloc(batch_size * sizeof(float));
    ssm->rms_grad = (float*)malloc(batch_size * state_dim * sizeof(float));

    // Allocate temporary buffers
    ssm->temp_state = (float*)malloc(batch_size * state_dim * sizeof(float));
    ssm->temp_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    ssm->A_stable = (float*)malloc(state_dim * state_dim * sizeof(float));
    
    return ssm;
}

// ---------------------------------------------------------------------
// Function: forward_pass
// Computes the forward pass:
//   Compute A_stable from A
//   pre_state = A_stable * state + B * X
//   norm_state = RMSNorm(pre_state)
//   next_state = swish(norm_state)
//   predictions = C * next_state + D * X
// Updates the internal state to next_state.
// ---------------------------------------------------------------------
void forward_pass(SSM* ssm, float* X) {
    const float alpha = 1.0f, beta = 0.0f;

    // Compute stable A from A for this forward pass
    compute_stable_A(ssm->A_stable, ssm->A, ssm->state_dim);

    // Compute pre_state = A_stable * state
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->state_dim,
                alpha, ssm->state, ssm->state_dim,
                ssm->A_stable, ssm->state_dim,
                beta, ssm->pre_state, ssm->state_dim);

    // Add input contribution: pre_state += B * X
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->state_dim, ssm->input_dim,
                alpha, X, ssm->input_dim,
                ssm->B, ssm->state_dim,
                alpha, ssm->pre_state, ssm->state_dim); // Add to existing pre_state

    // Apply RMS Normalization
    // Step 1: Calculate squared norms per row
    rms_norm_squared(ssm->rms_denom, ssm->pre_state, ssm->batch_size, ssm->state_dim);
    
    // Step 2: Apply normalization with weights
    rms_norm(ssm->norm_state, ssm->pre_state, ssm->rms_denom, ssm->rms_weight, 
             ssm->rms_eps, ssm->batch_size, ssm->state_dim);

    // Apply swish activation: next_state = swish(norm_state)
    int total_state = ssm->batch_size * ssm->state_dim;
    swish_forward(ssm->next_state, ssm->norm_state, total_state);
    
    // Compute predictions = C * next_state
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->output_dim, ssm->state_dim,
                alpha, ssm->next_state, ssm->state_dim,
                ssm->C, ssm->output_dim,
                beta, ssm->predictions, ssm->output_dim);
                             
    // Add direct feedthrough: predictions += D * X
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ssm->batch_size, ssm->output_dim, ssm->input_dim,
                alpha, X, ssm->input_dim,
                ssm->D, ssm->output_dim,
                alpha, ssm->predictions, ssm->output_dim); // Add to existing predictions
    
    // Update internal state: state = next_state
    memcpy(ssm->state, ssm->next_state, ssm->batch_size * ssm->state_dim * sizeof(float));
}

// ---------------------------------------------------------------------
// Function: calculate_loss
// Computes the Mean Squared Error loss between predictions and targets.
// ---------------------------------------------------------------------
float calculate_loss(SSM* ssm, float* y) {
    int size = ssm->batch_size * ssm->output_dim;
    mse_loss(ssm->error, ssm->predictions, y, size);
    
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += ssm->error[i] * ssm->error[i];
    }
    return loss / size;
}

// ---------------------------------------------------------------------
// Function: zero_gradients
// Clears the gradient arrays.
// ---------------------------------------------------------------------
void zero_gradients(SSM* ssm) {
    int size_A = ssm->state_dim * ssm->state_dim * sizeof(float);
    int size_B = ssm->state_dim * ssm->input_dim * sizeof(float);
    int size_C = ssm->output_dim * ssm->state_dim * sizeof(float);
    int size_D = ssm->output_dim * ssm->input_dim * sizeof(float);
    int size_rms = ssm->state_dim * sizeof(float);
    
    memset(ssm->A_grad, 0, size_A);
    memset(ssm->B_grad, 0, size_B);
    memset(ssm->C_grad, 0, size_C);
    memset(ssm->D_grad, 0, size_D);
    memset(ssm->rms_weight_grad, 0, size_rms);
}

// ---------------------------------------------------------------------
// Function: backward_pass
// Computes gradients through the network using the chain rule,
// including through the RMS normalization layer.
// ---------------------------------------------------------------------
void backward_pass(SSM* ssm, float* X) {
    const float alpha = 1.0f, beta = 0.0f;

    // Gradient for C: C_grad = error * (next_state)^T
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->output_dim, ssm->batch_size,
                alpha, ssm->next_state, ssm->state_dim,
                ssm->error, ssm->output_dim,
                beta, ssm->C_grad, ssm->output_dim);
                             
    // Gradient for D: D_grad = error * (X)^T
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->output_dim, ssm->batch_size,
                alpha, X, ssm->input_dim,
                ssm->error, ssm->output_dim,
                beta, ssm->D_grad, ssm->output_dim);
                             
    // Compute state error: state_error = C^T * error
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                ssm->batch_size, ssm->state_dim, ssm->output_dim,
                alpha, ssm->error, ssm->output_dim,
                ssm->C, ssm->output_dim,
                beta, ssm->state_error, ssm->state_dim);
                             
    // Apply swish backward: modify state_error in place
    int total_state = ssm->batch_size * ssm->state_dim;
    swish_backward(ssm->state_error, ssm->norm_state, ssm->next_state, total_state);
    
    // Backpropagate through RMS normalization
    // Use temp_state to store pre_state gradient
    rms_norm_backward(ssm->temp_state, ssm->rms_weight_grad,
                     ssm->state_error, ssm->pre_state,
                     ssm->rms_denom, ssm->rms_weight,
                     ssm->rms_eps, ssm->batch_size, ssm->state_dim);
    
    // Copy pre_state gradient for further backpropagation
    memcpy(ssm->state_error, ssm->temp_state, ssm->batch_size * ssm->state_dim * sizeof(float));
                                                      
    // First compute gradient for A_stable: temp = state_error * (state)^T
    // Reuse A_stable buffer for storing this gradient temporarily
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->state_dim, ssm->state_dim, ssm->batch_size,
                alpha, ssm->state, ssm->state_dim,
                ssm->state_error, ssm->state_dim,
                beta, ssm->A_stable, ssm->state_dim);
                             
    // Convert A_stable gradient to A gradient
    compute_A_grad_from_stable_grad(ssm->A_grad, ssm->A_stable, ssm->A, ssm->state_dim);
                             
    // Gradient for B: B_grad = state_error * (X)^T
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                ssm->input_dim, ssm->state_dim, ssm->batch_size,
                alpha, X, ssm->input_dim,
                ssm->state_error, ssm->state_dim,
                beta, ssm->B_grad, ssm->state_dim);
}

// ---------------------------------------------------------------------
// Function: update_weights
// Uses the AdamW optimizer to update each weight matrix
// ---------------------------------------------------------------------
void update_weights(SSM* ssm, float learning_rate) {
    ssm->adam_t++; // Increment time step
    float bias_correction1 = 1.0f - powf(ssm->beta1, (float)ssm->adam_t);
    float bias_correction2 = 1.0f - powf(ssm->beta2, (float)ssm->adam_t);

    int size_A = ssm->state_dim * ssm->state_dim;
    int size_B = ssm->state_dim * ssm->input_dim;
    int size_C = ssm->output_dim * ssm->state_dim;
    int size_D = ssm->output_dim * ssm->input_dim;
    int size_rms = ssm->state_dim;

    adamw_update(ssm->A, ssm->A_grad, ssm->A_m, ssm->A_v,
                size_A, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
                learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update(ssm->B, ssm->B_grad, ssm->B_m, ssm->B_v,
                size_B, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
                learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update(ssm->C, ssm->C_grad, ssm->C_m, ssm->C_v,
                size_C, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
                learning_rate, ssm->batch_size, bias_correction1, bias_correction2);

    adamw_update(ssm->D, ssm->D_grad, ssm->D_m, ssm->D_v,
                size_D, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
                learning_rate, ssm->batch_size, bias_correction1, bias_correction2);
                
    adamw_update(ssm->rms_weight, ssm->rms_weight_grad, ssm->rms_weight_m, ssm->rms_weight_v,
                size_rms, ssm->beta1, ssm->beta2, ssm->epsilon, ssm->weight_decay,
                learning_rate, ssm->batch_size, bias_correction1, bias_correction2);
}

// ---------------------------------------------------------------------
// Function: free_ssm
// Frees all allocated memory.
// ---------------------------------------------------------------------
void free_ssm(SSM* ssm) {
    // Free memory
    free(ssm->A);
    free(ssm->B);
    free(ssm->C);
    free(ssm->D);
    free(ssm->rms_weight);
    free(ssm->A_grad);
    free(ssm->B_grad);
    free(ssm->C_grad);
    free(ssm->D_grad);
    free(ssm->rms_weight_grad);
    free(ssm->A_m);
    free(ssm->A_v);
    free(ssm->B_m);
    free(ssm->B_v);
    free(ssm->C_m);
    free(ssm->C_v);
    free(ssm->D_m);
    free(ssm->D_v);
    free(ssm->rms_weight_m);
    free(ssm->rms_weight_v);
    free(ssm->state);
    free(ssm->next_state);
    free(ssm->pre_state);
    free(ssm->norm_state);
    free(ssm->predictions);
    free(ssm->error);
    free(ssm->state_error);
    free(ssm->rms_denom);
    free(ssm->rms_grad);
    free(ssm->temp_state);
    free(ssm->temp_output);
    free(ssm->A_stable);

    free(ssm);
}

// ---------------------------------------------------------------------
// Function: save_model
// Saves the model weights to a binary file.
// ---------------------------------------------------------------------
void save_ssm(SSM* ssm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }

    // Write dimensions
    fwrite(&ssm->input_dim, sizeof(int), 1, file);
    fwrite(&ssm->state_dim, sizeof(int), 1, file);
    fwrite(&ssm->output_dim, sizeof(int), 1, file);
    fwrite(&ssm->batch_size, sizeof(int), 1, file);
    
    // Write Adam hyperparameters
    fwrite(&ssm->beta1, sizeof(float), 1, file);
    fwrite(&ssm->beta2, sizeof(float), 1, file);
    fwrite(&ssm->epsilon, sizeof(float), 1, file);
    fwrite(&ssm->weight_decay, sizeof(float), 1, file);
    fwrite(&ssm->adam_t, sizeof(int), 1, file);
    fwrite(&ssm->rms_eps, sizeof(float), 1, file);

    // Write weight matrices to file
    fwrite(ssm->A, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->rms_weight, sizeof(float), ssm->state_dim, file);
    
    // Write Adam state to file
    fwrite(ssm->A_m, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->A_v, sizeof(float), ssm->state_dim * ssm->state_dim, file);
    fwrite(ssm->B_m, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->B_v, sizeof(float), ssm->state_dim * ssm->input_dim, file);
    fwrite(ssm->C_m, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->C_v, sizeof(float), ssm->output_dim * ssm->state_dim, file);
    fwrite(ssm->D_m, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->D_v, sizeof(float), ssm->output_dim * ssm->input_dim, file);
    fwrite(ssm->rms_weight_m, sizeof(float), ssm->state_dim, file);
    fwrite(ssm->rms_weight_v, sizeof(float), ssm->state_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: load_model
// Loads the model weights from a binary file and initializes a new SSM.
// ---------------------------------------------------------------------
SSM* load_ssm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }

    int input_dim, state_dim, output_dim, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&state_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    SSM* ssm = init_ssm(input_dim, state_dim, output_dim, batch_size);
    
    // Read Adam hyperparameters
    fread(&ssm->beta1, sizeof(float), 1, file);
    fread(&ssm->beta2, sizeof(float), 1, file);
    fread(&ssm->epsilon, sizeof(float), 1, file);
    fread(&ssm->weight_decay, sizeof(float), 1, file);
    fread(&ssm->adam_t, sizeof(int), 1, file);
    fread(&ssm->rms_eps, sizeof(float), 1, file);

    // Read weight matrices
    fread(ssm->A, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D, sizeof(float), output_dim * input_dim, file);
    fread(ssm->rms_weight, sizeof(float), state_dim, file);
    
    // Read Adam state
    fread(ssm->A_m, sizeof(float), state_dim * state_dim, file);
    fread(ssm->A_v, sizeof(float), state_dim * state_dim, file);
    fread(ssm->B_m, sizeof(float), state_dim * input_dim, file);
    fread(ssm->B_v, sizeof(float), state_dim * input_dim, file);
    fread(ssm->C_m, sizeof(float), output_dim * state_dim, file);
    fread(ssm->C_v, sizeof(float), output_dim * state_dim, file);
    fread(ssm->D_m, sizeof(float), output_dim * input_dim, file);
    fread(ssm->D_v, sizeof(float), output_dim * input_dim, file);
    fread(ssm->rms_weight_m, sizeof(float), state_dim, file);
    fread(ssm->rms_weight_v, sizeof(float), state_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ssm;
}

#endif