// tee_ta_gnn_nonlinear.c
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "teegnn_ta.h"
#include "blocked_edge_list.h"
#include "crypto.h"
#include "csprng_adapter_c.h"
#include "scan.h"
#include "tee_api_defines.h"
#include "tee_api_types.h"

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <tee_ta_api.h> 

#define INDEX(i, j) (i * cols + j)

typedef struct {
    uint32_t rows;
    uint32_t cols;
    size_t capacity;
    double* data;  // row-major
} tee_matrix_t;

// Subtly Designed Invertible Matrix
typedef struct SDIM {
    int n;
    size_t capacity;
    int32_t *perm;
    int32_t *value;
    int32_t *h;
} SDIM;

typedef struct {
    teegnn_random_engine_t rng_data;
    teegnn_random_engine_t rng_model;
    teegnn_random_engine_t rng_temp;
    
    uint8_t key[2][TEEGNN_AES128_KEY_LEN];
    uint8_t temp_key[TEEGNN_AES128_KEY_LEN];

    uint32_t row_block_size;
    uint32_t edge_block_size;
    uint32_t max_nodes;
    uint32_t feature_dim;
    uint32_t hidden_dim;
    uint32_t class_dim;
    uint32_t max_cols;
    double sum;
    double factor;
    double* col_sum;
    double* col_sum_out;
    double* temp_row;
    uint8_t *edge_payload;
    size_t edge_payload_capacity;
    SDIM temp_SDIM[2];
    tee_matrix_t temp_matrix;

    bool initialized;
} gnn_context_t;

static gnn_context_t global_ctx;
static gnn_context_t* ctx = NULL;

static size_t max_size(size_t a, size_t b) {
    return a > b ? a : b;
}

static double Exp(double x) {
    if (x <= 1e-15 && x >= -(1e-15)) return 1.0;
    
    int is_negative = 0;
    if (x < 0.0) {
        is_negative = 1;
        x = -x;
    }
    
    int integer_part = (int)x;
    double fractional_part = x - integer_part;
    
    double exp_integer = 1.0;
    double e = 2.71828182845904523536;
    for (int i = integer_part; i; i >>= 1) {
        if (i & 1) exp_integer *= e;
        e *= e;
    }
    
    double exp_fractional = 1.0;
    double term = 1.0;
    for (int n = 1; n < 30; n++) {
        term *= fractional_part / n;
        exp_fractional += term;
        if (term < 1e-15) break;
    }
    
    double result = exp_integer * exp_fractional;
    if (is_negative) {
        return 1.0 / result;
    }
    
    return result;
}

static TEE_Result checked_matrix_elements(uint32_t rows, uint32_t cols, size_t *out) {
    size_t total;

    if (out == NULL || rows == 0U || cols == 0U) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if ((size_t)rows > SIZE_MAX / (size_t)cols) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    total = (size_t)rows * (size_t)cols;
    if (total > SIZE_MAX / sizeof(double)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    *out = total;
    return TEE_SUCCESS;
}

static TEE_Result alloc_matrix(tee_matrix_t* mat, size_t capacity) {
    size_t bytes;

    if (mat == NULL || capacity == 0U || capacity > SIZE_MAX / sizeof(double)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    bytes = capacity * sizeof(double);
    mat->data = TEE_Malloc(bytes, TEE_MALLOC_FILL_ZERO);
    if (mat->data == NULL) {
        mat->rows = 0;
        mat->cols = 0;
        mat->capacity = 0;
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->capacity = capacity;
    return TEE_SUCCESS;
}

static TEE_Result prepare_matrix(tee_matrix_t* mat, uint32_t rows, uint32_t cols) {
    size_t total;
    TEE_Result res = checked_matrix_elements(rows, cols, &total);
    if (res != TEE_SUCCESS) {
        return res;
    }
    if (mat == NULL || mat->data == NULL || total > mat->capacity) {
        return TEE_ERROR_BAD_STATE;
    }
    mat->rows = rows;
    mat->cols = cols;
    TEE_MemFill(mat->data, 0, total * sizeof(double));
    return TEE_SUCCESS;
}

static void free_matrix(tee_matrix_t* mat) {
    if (mat->data != NULL) {
        TEE_Free(mat->data);
        mat->data = NULL;
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->capacity = 0;
}

static void free_sdim(SDIM *sdim) {
    if (sdim != NULL) {
        TEE_Free(sdim->perm);
        TEE_Free(sdim->value);
        TEE_Free(sdim->h);
        sdim->n = 0;
        sdim->capacity = 0;
        sdim->perm = NULL;
        sdim->value = NULL;
        sdim->h = NULL;
    }
}

static TEE_Result alloc_sdim(SDIM *sdim, size_t capacity) {
    if (sdim == NULL || capacity == 0U || capacity > (size_t)INT32_MAX) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (capacity > SIZE_MAX / sizeof(int32_t)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    sdim->perm = TEE_Malloc(capacity * sizeof(int32_t), 0);
    sdim->value = TEE_Malloc(capacity * sizeof(int32_t), 0);
    sdim->h = TEE_Malloc(capacity * sizeof(int32_t), 0);
    if (sdim->perm == NULL || sdim->value == NULL || sdim->h == NULL) {
        free_sdim(sdim);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    sdim->n = 0;
    sdim->capacity = capacity;
    return TEE_SUCCESS;
}

static void free_context_buffers(gnn_context_t *context) {
    if (context == NULL) {
        return;
    }
    for (size_t i = 0; i < 2; ++i) {
        free_sdim(&context->temp_SDIM[i]);
    }
    free_matrix(&context->temp_matrix);
    if (context->col_sum != NULL) {
        TEE_Free(context->col_sum);
        context->col_sum = NULL;
    }
    if (context->col_sum_out != NULL) {
        TEE_Free(context->col_sum_out);
        context->col_sum_out = NULL;
    }
    if (context->temp_row != NULL) {
        TEE_Free(context->temp_row);
        context->temp_row = NULL;
    }
    if (context->edge_payload != NULL) {
        TEE_Free(context->edge_payload);
        context->edge_payload = NULL;
    }
    TEE_MemFill(context->key, 0, sizeof(context->key));
    TEE_MemFill(context->temp_key, 0, sizeof(context->temp_key));
    context->edge_payload_capacity = 0;
    context->row_block_size = 0;
    context->edge_block_size = 0;
    context->max_nodes = 0;
    context->feature_dim = 0;
    context->hidden_dim = 0;
    context->class_dim = 0;
    context->max_cols = 0;
}

static TEE_Result generate_sdim(teegnn_random_engine_t *engine, size_t n, SDIM *sdim) {
    int rc;

    if (engine == NULL || sdim == NULL || n == 0 || n > (size_t)INT32_MAX) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (sdim->perm == NULL || sdim->value == NULL || sdim->h == NULL || n > sdim->capacity) {
        return TEE_ERROR_BAD_STATE;
    }

    for (size_t attempt = 0; attempt < 32; ++attempt) {
        double denominator = 1.0;
        sdim->n = (int)n;

        for (size_t i = 0; i < n; ++i) {
            int32_t value;

            sdim->perm[i] = (int32_t)i;
            rc = teegnn_random_nonzero_scale(engine, &value);
            if (rc != CSPRNG_OK) {
                return TEE_ERROR_BAD_STATE;
            }
            sdim->value[i] = value;
        }

        for (size_t i = n - 1; i > 0; --i) {
            uint32_t j;
            int32_t tmp;

            rc = teegnn_random_uniform_index(engine, (uint32_t)i + 1U, &j);
            if (rc != CSPRNG_OK) {
                return TEE_ERROR_BAD_STATE;
            }
            tmp = sdim->perm[i];
            sdim->perm[i] = sdim->perm[j];
            sdim->perm[j] = tmp;
        }

        for (size_t i = 0; i < n; ++i) {
            int32_t value;

            rc = teegnn_random_nonzero_scale(engine, &value);
            if (rc != CSPRNG_OK) {
                return TEE_ERROR_BAD_STATE;
            }
            sdim->h[i] = value;
            denominator += value / (double)sdim->value[i];
        }

        if (denominator <= -1e-8 || denominator >= 1e-8) {
            return TEE_SUCCESS;
        }
    }

    sdim->n = 0;
    return TEE_ERROR_BAD_STATE;
}

static TEE_Result allocate_context_buffers(
    gnn_context_t *context,
    uint32_t feature_dim,
    uint32_t hidden_dim,
    uint32_t num_nodes,
    uint32_t class_dim,
    uint32_t row_block_size,
    uint32_t edge_block_size
) {
    TEE_Result res;
    teegnn_status_t st;
    size_t max_cols = max_size((size_t)hidden_dim, (size_t)class_dim);
    size_t sdim0_capacity = max_size((size_t)feature_dim, (size_t)num_nodes);
    size_t sdim1_capacity = max_cols;
    size_t w1_elements = 0;
    size_t block_elements = 0;
    size_t matrix_capacity = 0;
    size_t edge_payload_capacity = 0;

    if (context == NULL || max_cols == 0U || row_block_size == 0U || edge_block_size == 0U) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (max_cols > SIZE_MAX / sizeof(double)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = checked_matrix_elements(feature_dim, hidden_dim, &w1_elements);
    if (res != TEE_SUCCESS) {
        return res;
    }
    res = checked_matrix_elements(row_block_size, (uint32_t)max_cols, &block_elements);
    if (res != TEE_SUCCESS) {
        return res;
    }
    matrix_capacity = max_size(w1_elements, block_elements);

    st = edge_block_payload_size(edge_block_size, &edge_payload_capacity);
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = alloc_sdim(&context->temp_SDIM[0], sdim0_capacity);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = alloc_sdim(&context->temp_SDIM[1], sdim1_capacity);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = alloc_matrix(&context->temp_matrix, matrix_capacity);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    context->col_sum = TEE_Malloc(max_cols * sizeof(double), TEE_MALLOC_FILL_ZERO);
    context->col_sum_out = TEE_Malloc(max_cols * sizeof(double), TEE_MALLOC_FILL_ZERO);
    context->temp_row = TEE_Malloc(max_cols * sizeof(double), 0);
    context->edge_payload = TEE_Malloc(edge_payload_capacity, 0);
    if (context->col_sum == NULL || context->col_sum_out == NULL ||
        context->temp_row == NULL || context->edge_payload == NULL) {
        res = TEE_ERROR_OUT_OF_MEMORY;
        goto fail;
    }

    context->row_block_size = row_block_size;
    context->edge_block_size = edge_block_size;
    context->max_nodes = num_nodes;
    context->feature_dim = feature_dim;
    context->hidden_dim = hidden_dim;
    context->class_dim = class_dim;
    context->max_cols = (uint32_t)max_cols;
    context->edge_payload_capacity = edge_payload_capacity;
    return TEE_SUCCESS;

fail:
    free_context_buffers(context);
    return res;
}

// matrices are in row-major order.
static TEE_Result sdim_mask(double *in, double *out, SDIM *L, SDIM *R) {
    size_t rows = L->n;
    size_t cols = R->n;
    double factor = 1.0;
    double sum = 0.0;

    double row_sum = 0.0;
    double* col_sum = ctx->col_sum;
    if (col_sum == NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            col_sum[j] += in[INDEX(i, j)];
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        factor += (double)(R->h[j]) / R->value[j];
        sum += col_sum[R->perm[j]] * R->h[j] / R->value[j];
    }
    sum /= factor;

    for (size_t i = 0; i < rows; ++i) {
        size_t n_i = L->perm[i];
        row_sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            row_sum += in[INDEX(n_i, R->perm[j])] / R->value[j] * R->h[j];
        }
        row_sum /= factor;
        for (size_t j = 0; j < cols; ++j) {
            size_t n_j = R->perm[j];
            out[INDEX(i, j)] = in[INDEX(n_i, n_j)] / R->value[j] * L->value[i] + 
                                      col_sum[n_j] / R->value[j] * L->h[i]     -
                                           row_sum / R->value[j] * L->value[i] -
                                               sum / R->value[j] * L->h[i];
        }
    }

    return TEE_SUCCESS;
}

static void apply_relu(tee_matrix_t* mat) {
    size_t total = mat->rows * mat->cols;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = mat->data[i] > 0 ? mat->data[i] : 0;
    }
}

static void apply_softmax(tee_matrix_t* mat) {
    size_t rows = mat->rows;
    size_t cols = mat->cols;
    for (size_t i = 0; i < rows; i++) {
        double max_val = mat->data[INDEX(i, 0)];
        for (size_t j = 1; j < cols; j++) {
            double val = mat->data[INDEX(i, j)];
            if (val > max_val) max_val = val;
        }
        
        double sum = 0;
        for (size_t j = 0; j < cols; j++) {
            double val = mat->data[INDEX(i, j)];
            mat->data[INDEX(i, j)] = Exp(val - max_val);
            sum += mat->data[INDEX(i, j)];
        }
        
        for (size_t j = 0; j < cols; j++) {
            mat->data[INDEX(i, j)] /= sum;
        }
    }
}

// stream scan for unmasked Y
static void get_row(const double* Y, uint32_t i, SDIM *L, SDIM *R) {
    size_t rows = L->n;
    size_t cols = R->n;
    double row_sum = 0.0;
    for (size_t j = 0; j < cols; ++j) {
        row_sum += Y[INDEX(i, j)] / L->value[i] * R->h[j];
    }
    for (size_t j = 0; j < cols; ++j) {
        int n_j = R->perm[j];
        ctx->temp_row[n_j] = Y[INDEX(i, j)] / L->value[i] * R->value[j] + row_sum -
                            ctx->col_sum[j] / L->value[i] * L->h[i]     -
                                   ctx->sum / L->value[i] * L->h[i];
    }
}

static TEE_Result blocked_matrix_encrypt(
    tee_matrix_t *Z,
    uint32_t rows,
    uint32_t row_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *ciphertext    
) {
    const uint32_t row_block_size = Z->rows;
    const uint32_t cols = Z->cols;
    uint32_t blocks = 0;
    size_t block_payload_len = 0;
    size_t block_stride = 0;
    teegnn_status_t st = matrix_block_layout(
        row_block_size,
        rows,
        cols,
        &blocks,
        &block_payload_len,
        &block_stride
    );
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_STATE;
    }
    if (blocks == 0) {
        return TEE_SUCCESS;
    }
    if (ciphertext == NULL || (block_payload_len > 0 && Z == NULL)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    uint8_t *block_base = ciphertext + (size_t)row_block_id * block_stride;
    uint8_t *nonce = block_base;
    uint8_t *tag = nonce + TEEGNN_GCM_NONCE_LEN;
    uint8_t *block_ciphertext = tag + TEEGNN_GCM_TAG_LEN;

    uint8_t aad[TEEGNN_MATRIX_BLOCK_AAD_LEN];
    size_t aad_len = 0;
    st = derive_matrix_nonce_simple(
        row_block_size,
        rows,
        cols,
        blocks,
        row_block_id,
        nonce
    );
    if (st == TEEGNN_OK) {
        st = build_matrix_block_aad(
            row_block_size,
            rows,
            cols,
            blocks,
            row_block_id,
            aad,
            sizeof(aad),
            &aad_len
        );
    }
    if (st == TEEGNN_OK) {
        st = teegnn_aes_gcm_encrypt(
            key,
            key_len,
            nonce,
            aad,
            aad_len,
            (uint8_t*)Z->data,
            block_payload_len,
            block_payload_len == 0 ? NULL : block_ciphertext,
            tag
        );
    }
    if (st != TEEGNN_OK) {
        return st;
    }
    return TEEGNN_OK;
}

static TEE_Result blocked_edge_list_stream_scan(
    const EncryptedBlockedEdgeList *enc,
    const uint8_t *key,
    size_t key_len,
    const double *Y,
    tee_matrix_t *Z,
    uint32_t layer_idx,
    uint8_t *ciphertext
) {
    if (enc == NULL) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    teegnn_status_t st = validate_edge_header_for_scan(&enc->header);
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_STATE;
    }
    st = validate_edge_object_layout_for_scan(enc);
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_STATE;
    }
    if (enc->header.n_nodes > 0 && (Y == NULL || Z == NULL)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    const size_t row_block_size = ctx->row_block_size;
    const size_t rows = enc->header.n_nodes;
    const size_t cols = Z->cols;
    const size_t total = row_block_size * cols;
    if (total > 0) {
        TEE_MemFill(Z->data, 0, total * sizeof(double));
    }

    if (enc->header.m_edges == 0) {
        return TEE_SUCCESS;
    }

    size_t block_payload_len = 0;
    st = edge_block_payload_size(enc->header.block_size, &block_payload_len);
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_STATE;
    }
    if (ctx->edge_payload == NULL || block_payload_len > ctx->edge_payload_capacity) {
        return TEE_ERROR_BAD_STATE;
    }
    uint8_t *payload = ctx->edge_payload;

    size_t current_row_block = 0;
    size_t current_row = 0;
    get_row(Y, current_row, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    for (uint32_t block_id = 0; block_id < enc->header.num_blocks; ++block_id) {
        st = decrypt_edge_blob_for_scan(
            &enc->header,
            enc,
            TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id),
            block_id,
            key,
            key_len,
            payload,
            block_payload_len
        );
        if (st != TEEGNN_OK) {
            return TEE_ERROR_BAD_STATE;
        }

        const uint8_t *dsts = payload;
        const uint8_t *srcs = dsts + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *values = srcs + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *valid = values + (size_t)enc->header.block_size * sizeof(double);

        for (uint32_t t = 0; t < enc->header.block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * enc->header.block_size + t;
            const uint8_t is_valid = valid[t];
            if (is_valid > 1U) {
                return TEE_ERROR_BAD_FORMAT;
            }
            if (global_pos >= enc->header.m_edges) {
                const uint32_t dst = load_u32_slot(dsts, t);
                const uint32_t src = load_u32_slot(srcs, t);
                const double value = load_double_slot(values, t);
                if (is_valid != 0U || dst != 0U || src != 0U || value != 0.0) {
                    return TEE_ERROR_BAD_FORMAT;
                }
                continue;
            }
            if (is_valid != 1U) {
                return TEE_ERROR_BAD_FORMAT;
            }

            const uint32_t dst = load_u32_slot(dsts, t);
            const uint32_t src = load_u32_slot(srcs, t);
            if (dst / row_block_size > current_row_block) {
                if (layer_idx == 0) {
                    apply_relu(Z);
                } else if (layer_idx == 1) {
                    apply_softmax(Z);
                } else {
                    return TEE_ERROR_BAD_PARAMETERS;
                }
                // update col_sum_out
                for (uint32_t i = 0; i < row_block_size; ++i) {
                    for (uint32_t j = 0; j < cols; ++j) {
                        ctx->col_sum_out[j] += Z->data[INDEX(i, j)];
                    }
                }
                // write encrypted row block to REE
                blocked_matrix_encrypt(
                    Z, 
                    rows, 
                    current_row_block, 
                    ctx->temp_key, 
                    TEEGNN_AES128_KEY_LEN, 
                    ciphertext
                );

                // reset Z
                TEE_MemFill(Z->data, 0, total * sizeof(double));

                // reset current_row index to 0
                current_row = 0;
                get_row(Y, current_row, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
                current_row_block++;
            }
            while (current_row < src) {
                current_row++;
                get_row(Y, current_row, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
            }
            if (src >= enc->header.n_nodes || dst >= enc->header.n_nodes) {
                return TEE_ERROR_BAD_PARAMETERS;
            }

            const double value = load_double_slot(values, t);
            for (uint32_t f = 0; f < cols; ++f) {
                Z->data[INDEX(dst % row_block_size, f)] += value * ctx->temp_row[f];
            }
        }

    }

    // last row block
    if (layer_idx == 0) {
        apply_relu(Z);
    } else if (layer_idx == 1) {
        apply_softmax(Z);
    } else {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    // update col_sum_out
    for (uint32_t i = 0; i < row_block_size; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            ctx->col_sum_out[j] += Z->data[INDEX(i, j)];
        }
    }
    blocked_matrix_encrypt(
        Z, 
        rows, 
        current_row_block, 
        ctx->temp_key, 
        TEEGNN_AES128_KEY_LEN, 
        ciphertext
    );

    return TEEGNN_OK;
}

static TEE_Result blocked_matrix_decrypt(
    tee_matrix_t *Z,
    double *out,
    SDIM *L,
    SDIM *R,
    const uint8_t *ciphertext,
    uint32_t layer_idx
) {
    const uint32_t row_block_size = Z->rows;
    const uint32_t rows = L->n;
    const uint32_t cols = Z->cols;
    const uint8_t* key = ctx->temp_key;
    const uint32_t key_len = TEEGNN_AES128_KEY_LEN;
    double sum = 0.0;
    double factor = 1.0;
    uint32_t blocks = 0;
    size_t block_payload_len = 0;
    size_t block_stride = 0;
    teegnn_status_t st = matrix_block_layout(
        row_block_size,
        rows,
        cols,
        &blocks,
        &block_payload_len,
        &block_stride
    );
    if (st != TEEGNN_OK) {
        return TEE_ERROR_BAD_STATE;
    }

    if (blocks == 0) {
        return TEE_SUCCESS;
    }
    if (ciphertext == NULL || (block_payload_len > 0 && Z == NULL)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    for (size_t j = 0; j < cols; ++j) {
        factor += (double)(R->h[j]) / R->value[j];
        sum += ctx->col_sum_out[R->perm[j]] / R->value[j] * R->h[j] ;
    }
    sum /= factor;

    for (uint32_t row_block_id = 0; row_block_id < blocks; ++row_block_id) {
        uint8_t *payload = (uint8_t*)Z->data;

        const uint8_t *block_base = ciphertext + (size_t)row_block_id * block_stride;
        const uint8_t *nonce = block_base;
        const uint8_t *tag = nonce + TEEGNN_GCM_NONCE_LEN;
        const uint8_t *block_ciphertext = tag + TEEGNN_GCM_TAG_LEN;

        uint8_t aad[TEEGNN_MATRIX_BLOCK_AAD_LEN];
        size_t aad_len = 0;
        st = build_matrix_block_aad(
            row_block_size,
            rows,
            cols,
            blocks,
            row_block_id,
            aad,
            sizeof(aad),
            &aad_len
        );
        if (st == TEEGNN_OK) {
            st = teegnn_aes_gcm_decrypt(
                key,
                key_len,
                nonce,
                aad,
                aad_len,
                block_payload_len == 0 ? NULL : block_ciphertext,
                block_payload_len,
                tag,
                payload
            );
        }
        if (st != TEEGNN_OK) {
            return TEE_ERROR_BAD_STATE;
        }

        // write to REE
        uint32_t offset = row_block_id * row_block_size;
        for (size_t i = 0; i < row_block_size; ++i) {
            if (i + offset >= rows) {
                break;
            }
            double row_sum = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                row_sum += Z->data[INDEX(i, R->perm[j])] / R->value[j] * R->h[j];
            }
            row_sum /= factor;
            size_t global_i = i + offset;
            for (size_t j = 0; j < cols; ++j) {
                size_t n_j = R->perm[j];
                if (layer_idx == 1) {
                    out[INDEX(L->perm[global_i], j)] = Z->data[INDEX(i, j)];
                    continue;
                }
                out[INDEX(global_i, j)] = Z->data[INDEX(i, n_j)] / R->value[j] * L->value[global_i] + 
                                           ctx->col_sum_out[n_j] / R->value[j] * L->h[global_i]     -
                                                         row_sum / R->value[j] * L->value[global_i] -
                                                             sum / R->value[j] * L->h[global_i];
            }
        }
    }
    return TEEGNN_OK;
}

TEE_Result TA_CreateEntryPoint(void) {
    // DMSG("GNN Nonlinear TA created");
    printf("GNN Nonlinear TA created\n");
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void) {
    DMSG("GNN Nonlinear TA destroyed");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param params[4],
                                    void** sess_ctx) {
    (void)param_types;
    (void)params;

    if (ctx != NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    TEE_MemFill(&global_ctx, 0, sizeof(global_ctx));
    ctx = &global_ctx;
    *sess_ctx = (void*)ctx;

    DMSG("GNN Nonlinear session opened");
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void* sess_ctx) {
    (void)sess_ctx;

    if (ctx != NULL) {
        teegnn_random_engine_wipe(&ctx->rng_data);
        teegnn_random_engine_wipe(&ctx->rng_model);
        teegnn_random_engine_wipe(&ctx->rng_temp);
        free_context_buffers(ctx);
        TEE_MemFill(ctx, 0, sizeof(*ctx));
        ctx = NULL;
    }
    
    DMSG("GNN Nonlinear session closed");
}

static TEE_Result init_context(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,  // w1
        TEE_PARAM_TYPE_MEMREF_INPUT,  // seeds, keys, and block sizes
        TEE_PARAM_TYPE_VALUE_INPUT,   // feature_dim, hidden_dim
        TEE_PARAM_TYPE_VALUE_INPUT    // num_nodes, class_dim
    );
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (ctx == NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    TEE_Result res = TEE_SUCCESS;
    uint32_t feature_dim = params[2].value.a;
    uint32_t hidden_dim = params[2].value.b;
    uint32_t num_nodes = params[3].value.a;
    uint32_t class_dim = params[3].value.b;
    const size_t secrets_size = 2U * sizeof(uint32_t) + 2U * sizeof(uint64_t) +
                                2U * TEEGNN_AES128_KEY_LEN;
    size_t w1_elements = 0;
    size_t w1_bytes = 0;

    char label_data[] = "teegnn-data-owner";
    char label_model[] = "teegnn-model-owner";
    char label_temp[] = "teegnn-temp";
    uint32_t row_block_size = 0;
    uint32_t edge_block_size = 0;
    uint64_t seed_data = 0;
    uint64_t seed_model = 0;
    uint8_t* secrets = params[1].memref.buffer;
    size_t offset = 0;
    double *w1 = (double*)params[0].memref.buffer;

    if (w1 == NULL || secrets == NULL || feature_dim == 0U || hidden_dim == 0U ||
        num_nodes == 0U || class_dim == 0U) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (params[1].memref.size < secrets_size) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    res = checked_matrix_elements(feature_dim, hidden_dim, &w1_elements);
    if (res != TEE_SUCCESS) {
        return res;
    }
    w1_bytes = w1_elements * sizeof(double);
    if (params[0].memref.size < w1_bytes) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    free_context_buffers(ctx);
    ctx->initialized = false;

    // format: [row_block_size][edge_block_size][seed_data][seed_model][key1][key2]
    TEE_MemMove(&row_block_size, secrets + offset, sizeof(row_block_size));
    offset += sizeof(row_block_size);
    TEE_MemMove(&edge_block_size, secrets + offset, sizeof(edge_block_size));
    offset += sizeof(edge_block_size);
    TEE_MemMove(&seed_data, secrets + offset, sizeof(seed_data));
    offset += sizeof(seed_data);
    TEE_MemMove(&seed_model, secrets + offset, sizeof(seed_model));
    offset += sizeof(seed_model);

    if (row_block_size == 0U || row_block_size > num_nodes || edge_block_size == 0U) {
        res = TEE_ERROR_BAD_PARAMETERS;
        goto fail;
    }

    res = allocate_context_buffers(
        ctx,
        feature_dim,
        hidden_dim,
        num_nodes,
        class_dim,
        row_block_size,
        edge_block_size
    );
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    for (size_t i = 0; i < 2; ++i) {
        TEE_MemMove(ctx->key[i], secrets + offset, TEEGNN_AES128_KEY_LEN);
        offset += TEEGNN_AES128_KEY_LEN;
    }

    // initialize random engine
    if (teegnn_random_engine_init_with_label(&ctx->rng_data,  seed_data, 0,
                                            label_data, strlen(label_data)) != CSPRNG_OK ||
        teegnn_random_engine_init_with_label(&ctx->rng_model, seed_model, 0,
                                            label_model, strlen(label_model)) != CSPRNG_OK ||
        teegnn_random_engine_init_with_label(&ctx->rng_temp, 1234, 0,
                                            label_temp, strlen(label_temp)) != CSPRNG_OK) {
        res = TEE_ERROR_BAD_STATE;
        goto fail;
    }

    // sdim_mask for w1
    res = generate_sdim(&ctx->rng_data, feature_dim, &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    res = generate_sdim(&ctx->rng_temp, hidden_dim, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    res = prepare_matrix(&ctx->temp_matrix, feature_dim, hidden_dim);
    if (res != TEE_SUCCESS) {
        goto fail;
    }

    TEE_MemFill(ctx->col_sum, 0, hidden_dim * sizeof(double));
    res = sdim_mask(w1, ctx->temp_matrix.data, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        goto fail;
    }
    TEE_MemMove(w1, ctx->temp_matrix.data, w1_bytes);

    ctx->initialized = true;
    return TEE_SUCCESS;

fail:
    teegnn_random_engine_wipe(&ctx->rng_data);
    teegnn_random_engine_wipe(&ctx->rng_model);
    teegnn_random_engine_wipe(&ctx->rng_temp);
    free_context_buffers(ctx);
    ctx->initialized = false;
    return res;
}

// message passing and activation function
static TEE_Result secure_compute(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,    // masked matrix
        TEE_PARAM_TYPE_MEMREF_INPUT,    // Blocked Edge list
        TEE_PARAM_TYPE_MEMREF_INOUT,    // encrypted Z
        TEE_PARAM_TYPE_VALUE_INPUT      // num_nodes, current_dim
    );

    TEE_Result res;
    size_t y_elements = 0;
    size_t y_bytes = 0;
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (!ctx || !ctx->initialized) {
        return TEE_ERROR_BAD_STATE;
    }

    // parse parameters
    double* y = (double*)params[0].memref.buffer;
    EncryptedBlockedEdgeList* enc = (EncryptedBlockedEdgeList *)params[1].memref.buffer;
    uint8_t *ciphertext = params[2].memref.buffer;
    uint32_t rows = params[3].value.a;
    uint32_t cols = params[3].value.b;
    uint32_t layer;

    if (y == NULL || enc == NULL || ciphertext == NULL) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    layer = enc->header.layer_id;
    if (layer > 1U || rows != ctx->max_nodes || enc->header.n_nodes != rows ||
        cols == 0U || cols > ctx->max_cols) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if ((layer == 0U && cols != ctx->hidden_dim) ||
        (layer == 1U && cols != ctx->class_dim)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    res = checked_matrix_elements(rows, cols, &y_elements);
    if (res != TEE_SUCCESS) {
        return res;
    }
    y_bytes = y_elements * sizeof(double);
    if (params[0].memref.size < y_bytes) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // apply sdim unmask
    res = prepare_matrix(&ctx->temp_matrix, ctx->row_block_size, cols);
    if (res != TEE_SUCCESS) {
        return res;
    }

    if (layer == 0U) {
        res = generate_sdim(&ctx->rng_data, rows, &ctx->temp_SDIM[0]);
        if (res != TEE_SUCCESS) {
            return res;
        }
    } else {
        res = generate_sdim(&ctx->rng_model, cols, &ctx->temp_SDIM[1]);
        if (res != TEE_SUCCESS) {
            return res;
        }
    }

    SDIM* L = &ctx->temp_SDIM[0];
    SDIM* R = &ctx->temp_SDIM[1];
    TEE_MemFill(ctx->col_sum, 0, cols * sizeof(double));
    TEE_MemFill(ctx->col_sum_out, 0, cols * sizeof(double));
    TEE_MemFill(ctx->temp_row, 0, cols * sizeof(double));

    double* col_sum = ctx->col_sum;
    // 1. calculate the col sum in order
    ctx->sum = 0;
    ctx->factor = 1.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ctx->sum += y[INDEX(i, j)] / L->value[i] * R->h[j];
            col_sum[j] += y[INDEX(i, j)] / L->value[i] * R->value[j];
        }
        ctx->factor += (double)(L->h[i]) / L->value[i];
    }
    for (size_t j = 0; j < cols; ++j) {
        col_sum[j] /= ctx->factor;
    }
    ctx->sum /= ctx->factor;

    // 2. message passing and activation function
    TEE_GenerateRandom(ctx->temp_key, TEEGNN_AES128_KEY_LEN);
    res = blocked_edge_list_stream_scan(
        enc,
        ctx->key[layer],
        TEEGNN_AES128_KEY_LEN,
        y,
        &ctx->temp_matrix,
        layer,
        ciphertext
    );
    if (res != TEE_SUCCESS) {
        goto out;
    }

    // 3. decrypt from REE, then return masked matrix to REE
    res = generate_sdim(&ctx->rng_data, rows, &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        goto out;
    }
    res = generate_sdim(&ctx->rng_model, cols, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        goto out;
    }

    ctx->sum = 0;
    ctx->factor = 1.0;
    for (size_t j = 0; j < cols; ++j) {
        ctx->factor += (double)(R->h[j]) / R->value[j];
        ctx->sum += ctx->col_sum_out[R->perm[j]] * R->h[j] / R->value[j];
    }
    ctx->sum /= ctx->factor;

    res = blocked_matrix_decrypt(
        &ctx->temp_matrix,
        y,
        &ctx->temp_SDIM[0],
        &ctx->temp_SDIM[1],
        ciphertext,
        layer
    );

out:
    TEE_MemFill(ctx->temp_key, 0, sizeof(ctx->temp_key));
    return res;
}

static TEE_Result cleanup_context(uint32_t param_types, TEE_Param params[4]) {
    (void)param_types;
    (void)params;

    if (ctx) {
        ctx->initialized = false;
        ctx->sum = 0;
        ctx->factor = 0;
        TEE_MemFill(ctx->temp_key, 0, sizeof(ctx->temp_key));
    }

    return TEE_SUCCESS;
}

static TEE_Result get_debug_info(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_OUTPUT,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );
    
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    if (!ctx) {
        return TEE_ERROR_BAD_STATE;
    }
    
    char* buf = params[0].memref.buffer;
    size_t buf_size = params[0].memref.size;
    
    TEE_MemMove(buf, ctx->temp_SDIM[1].h, buf_size);
    
    return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void* sess_ctx, uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4]) {
    (void)sess_ctx;
    
    switch (cmd_id) {
        case TEEGNN_CMD_INIT_CONTEXT:
            return init_context(param_types, params);
        case TEEGNN_CMD_SECURE_COMPUTE:
            return secure_compute(param_types, params);
        case TEEGNN_CMD_FINALIZE_RESULT:
            return cleanup_context(param_types, params);
        case TEEGNN_CMD_GET_DEBUG_INFO:
            return get_debug_info(param_types, params);
        default:
            return TEE_ERROR_NOT_SUPPORTED;
    }
}
