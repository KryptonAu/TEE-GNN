// tee_ta_gnn_nonlinear.c
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "teegnn_ta.h"
#include "blocked_csc.h"
#include "csprng_adapter_c.h"
#include "scan.h"
#include "tee_api_defines.h"

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <tee_ta_api.h> 

#define INDEX(i, j) (i * cols + j)

/* 矩阵结构体，用于TEE内部表示 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    double* data;  // 按列主序存储
} tee_matrix_t;

// Scaled Permutation Matrix
typedef struct SPM {
    int n;
    int32_t *perm;
    int32_t *value;
} SPM;

// Subtly Designed Invertible Matrix
typedef struct SDIM {
    int n;
    int32_t *perm;
    int32_t *value;
    int32_t *h;
} SDIM;

typedef struct {
    /* 随机流 */
    teegnn_random_engine_t rng_data;
    teegnn_random_engine_t rng_model;
    teegnn_random_engine_t rng_temp;
    
    /* 参数 */
    uint8_t **key;

    /* 临时缓冲区 */
    double sum;
    double factor;
    tee_matrix_t temp_matrix;
    double* row_sum;
    double* col_sum;
    double* temp_row;
    SDIM temp_SDIM[2];

    bool initialized;
} gnn_context_t;

/* 全局上下文指针 */
static gnn_context_t* ctx = NULL;

/* 内部辅助函数 */
static double Exp(double x) {
    if (x <= 1e-15 && x >= -(1e-15)) return 1.0;
    
    int is_negative = 0;
    if (x < 0.0) {
        is_negative = 1;
        x = -x;
    }
    
    // 将x分解为整数部分和小数部分: exp(x) = exp(整数)*exp(小数)
    int integer_part = (int)x;
    double fractional_part = x - integer_part;
    
    // 计算exp(整数部分) - 通过连乘计算
    double exp_integer = 1.0;
    double e = 2.71828182845904523536;  // 自然常数e
    for (int i = integer_part; i; i >>= 1) {
        if (i & 1) exp_integer *= e;
        e *= e;
    }
    
    // 计算exp(小数部分) - 使用泰勒级数
    double exp_fractional = 1.0;
    double term = 1.0;
    for (int n = 1; n < 30; n++) {
        term *= fractional_part / n;
        exp_fractional += term;
        if (term < 1e-15) break;
    }
    
    double result = exp_integer * exp_fractional;
    // 如果是负数输入，返回倒数
    if (is_negative) {
        return 1.0 / result;
    }
    
    return result;
}

static TEE_Result init_matrix(tee_matrix_t* mat, uint32_t rows, uint32_t cols) {
    size_t total;
    size_t bytes;

    if (mat == NULL || rows == 0U || cols == 0U) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if ((size_t)rows > SIZE_MAX / (size_t)cols) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    total = (size_t)rows * (size_t)cols;
    if (total > SIZE_MAX / sizeof(double)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    bytes = total * sizeof(double);

    mat->data = TEE_Malloc(bytes, 0);
    if (mat->data == NULL) {
        mat->rows = 0;
        mat->cols = 0;
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    mat->rows = rows;
    mat->cols = cols;
    return TEE_SUCCESS;
}

static void free_matrix(tee_matrix_t* mat) {
    if (mat->data != NULL) {
        TEE_Free(mat->data);
        mat->data = NULL;
    }
    mat->rows = 0;
    mat->cols = 0;
}

static void free_spm(SPM *spm) {
    if (spm != 0) {
        TEE_Free(spm->perm);
        TEE_Free(spm->value);
        spm->n = 0;
        spm->perm = NULL;
        spm->value = NULL;
    }
}

static void free_sdim(SDIM *sdim) {
    if (sdim != 0) {
        TEE_Free(sdim->perm);
        TEE_Free(sdim->value);
        TEE_Free(sdim->h);
        sdim->n = 0;
        sdim->perm = NULL;
        sdim->value = NULL;
        sdim->h = NULL;
    }
}

static void free_context_buffers(gnn_context_t *context) {
    if (context == NULL) {
        return;
    }
    for (size_t i = 0; i < 2; ++i) {
        free_sdim(&context->temp_SDIM[i]);
    }
    free_matrix(&context->temp_matrix);
    if (context->row_sum != NULL) {
        TEE_Free(context->row_sum);
        context->row_sum = NULL;
    }
    if (context->col_sum != NULL) {
        TEE_Free(context->col_sum);
        context->col_sum = NULL;
    }
    if (context->temp_row != NULL) {
        TEE_Free(context->temp_row);
        context->temp_row = NULL;
    }
    if (context->key != NULL) {
        for (size_t i = 0; i < 2; ++i) {
            TEE_Free(context->key[i]);
        }
        TEE_Free(context->key);
        context->key = NULL;
    }
}

static TEE_Result generate_spm(teegnn_random_engine_t *engine, size_t n, SPM *spm) {
    int rc;

    if (engine == NULL || spm == NULL || n == 0 || n > (size_t)INT32_MAX) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    free_spm(spm);

    spm->n = (int)n;
    spm->perm = TEE_Malloc(n * sizeof(int32_t), 0);
    if (spm->perm == NULL) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    
    spm->value = TEE_Malloc(n * sizeof(int32_t), 0);
    if (spm->value == NULL) {
        free_spm(spm);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < n; ++i) {
        int32_t value;

        spm->perm[i] = i;
        rc = teegnn_random_nonzero_scale(engine, &value);
        if (rc != CSPRNG_OK) {
            free_spm(spm);
            return TEE_ERROR_BAD_STATE;
        }
        spm->value[i] = value;
    }

    for (size_t i = n - 1; i > 0; --i) {
        uint32_t j;
        int32_t tmp;

        rc = teegnn_random_uniform_index(engine, (uint32_t)i + 1U, &j);
        if (rc != CSPRNG_OK) {
            free_spm(spm);
            return TEE_ERROR_BAD_STATE;
        }
        tmp = spm->perm[i];
        spm->perm[i] = spm->perm[j];
        spm->perm[j] = tmp;
    }

    return TEE_SUCCESS;
}

static TEE_Result generate_sdim(teegnn_random_engine_t *engine, size_t n, SDIM *sdim) {
    SPM spm;
    int32_t *h;
    int rc;

    if (engine == NULL || sdim == NULL || n == 0 || n > (size_t)INT32_MAX) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    free_sdim(sdim);

    sdim->n = 0;
    sdim->perm = NULL;
    sdim->value = NULL;
    sdim->h = NULL;

    for (size_t attempt = 0; attempt < 32; ++attempt) {
        double denominator = 1.0;

        spm.n = 0;
        spm.perm = NULL;
        spm.value = NULL;
        h = NULL;

        TEE_Result res = generate_spm(engine, n, &spm);
        if (res != TEE_SUCCESS) {
            return res;
        }

        h = TEE_Malloc(n * sizeof(int32_t), 0);
        if (h == NULL) {
            free_spm(&spm);
            return TEE_ERROR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < n; ++i) {
            int32_t value;

            rc = teegnn_random_nonzero_scale(engine, &value);
            if (rc != CSPRNG_OK) {
                free_spm(&spm);
                TEE_Free(h);
                return TEE_ERROR_BAD_STATE;
            }
            h[i] = value;
            denominator += value / (double)spm.value[i];
        }

        if (denominator <= -1e-8 || denominator >= 1e-8) {
            sdim->n = n;
            sdim->perm = spm.perm;
            sdim->value = spm.value;
            sdim->h = h;
            return TEE_SUCCESS;
        }

        free_spm(&spm);
        TEE_Free(h);
    }

    return TEE_ERROR_BAD_STATE;
}

// matrices are in row-major order.
static TEE_Result sdim_mask(double *in, double *out, SDIM *L, SDIM *R) {
    size_t rows = L->n;
    size_t cols = R->n;
    double factor = 1.0;
    double sum = 0.0;

    double* row_sum = ctx->row_sum;
    double* col_sum = ctx->col_sum;
    if (row_sum == NULL || col_sum == NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            row_sum[i] += in[INDEX(i, R->perm[j])] / R->value[j] * R->h[j];
            col_sum[j] += in[INDEX(i, j)];
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        factor += (double)(R->h[j]) / R->value[j];
        sum += col_sum[R->perm[j]] * R->h[j] / R->value[j];
    }
    for (size_t i = 0; i < rows; ++i) {
        row_sum[i] /= factor;
    }
    sum /= factor;

    for (size_t i = 0; i < rows; ++i) {
        size_t n_i = L->perm[i];
        for (size_t j = 0; j < cols; ++j) {
            size_t n_j = R->perm[j];
            out[INDEX(i, j)] = in[INDEX(n_i, n_j)] / R->value[j] * L->value[i] + 
                                      col_sum[n_j] / R->value[j] * L->h[i]     -
                                      row_sum[n_i] / R->value[j] * L->value[i] -
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
    for (size_t i = 0; i < mat->rows; i++) {
        // 找最大值（数值稳定）
        double max_val = mat->data[i];
        for (size_t j = 1; j < mat->cols; j++) {
            double val = mat->data[i + j * mat->rows];
            if (val > max_val) max_val = val;
        }
        
        // 计算exp和sum
        double sum = 0;
        for (size_t j = 0; j < mat->cols; j++) {
            double val = mat->data[i + j * mat->rows];
            mat->data[i + j * mat->rows] = Exp(val - max_val);
            sum += mat->data[i + j * mat->rows];
        }
        
        // 归一化
        for (size_t j = 0; j < mat->cols; j++) {
            mat->data[i + j * mat->rows] /= sum;
        }
    }
}

static void get_row(const double* Y, uint32_t i, SDIM *L, SDIM *R) {
    size_t rows = L->n;
    size_t cols = R->n;
    for (size_t j = 0; j < cols; ++j) {
        int n_j = R->perm[j];
        ctx->temp_row[n_j] = Y[INDEX(i, j)] / L->value[i] * R->value[j] + ctx->row_sum[i] -
                            ctx->col_sum[j] / L->value[i] * L->h[i]     -
                                   ctx->sum / L->value[i] * L->h[i];
    }
}

teegnn_status_t blocked_csc_stream_scan(
    const EncryptedBlockedCSC *enc,
    const uint8_t *key,
    size_t key_len,
    const double *Y,
    uint32_t cols,
    double *Z
) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = validate_header_for_scan(&enc->header);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (cols > 0 && enc->header.n_nodes > 0 && (Y == NULL || Z == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    const size_t total = (size_t)enc->header.n_nodes * (size_t)cols;
    if (cols != 0 && total / cols != enc->header.n_nodes) {
        return TEEGNN_ERR_ALLOC;
    }
    if (total > 0) {
        memset(Z, 0, total * sizeof(double));
    }

    size_t col_payload_len = 0;
    st = col_ptr_payload_size(enc->header.n_nodes, &col_payload_len);
    if (st != TEEGNN_OK) {
        return st;
    }
    uint32_t *col_ptr = (uint32_t *)TEE_Malloc(col_payload_len * sizeof(uint32_t), 0);
    if (col_ptr == NULL) {
        return TEEGNN_ERR_ALLOC;
    }

    st = decrypt_blob_for_scan(
        &enc->header,
        enc,
        TEEGNN_BLOB_INDEX_COL_PTR,
        TEEGNN_COL_PTR_BLOCK_ID,
        key,
        key_len,
        (uint8_t *)col_ptr,
        col_payload_len
    );
    if (st != TEEGNN_OK) {
        TEE_Free(col_ptr);
        return st;
    }

    st = validate_col_ptr(col_ptr, enc->header.n_nodes, enc->header.nnz);
    if (st != TEEGNN_OK) {
        TEE_Free(col_ptr);
        return st;
    }

    if (enc->header.nnz == 0) {
        TEE_Free(col_ptr);
        return TEEGNN_OK;
    }

    size_t block_payload_len = 0;
    st = row_block_payload_size(enc->header.block_size, &block_payload_len);
    if (st != TEEGNN_OK) {
        TEE_Free(col_ptr);
        return st;
    }

    uint32_t current_col = 0;
    get_row(Y, 0, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    for (uint32_t block_id = 0; block_id < enc->header.num_blocks; ++block_id) {
        uint8_t *payload = (uint8_t *)TEE_Malloc(block_payload_len * sizeof(uint8_t), 0);
        if (payload == NULL) {
            TEE_Free(col_ptr);
            return TEEGNN_ERR_ALLOC;
        }

        st = decrypt_blob_for_scan(
            &enc->header,
            enc,
            TEEGNN_BLOB_INDEX_ROW_BLOCK(block_id),
            block_id,
            key,
            key_len,
            payload,
            block_payload_len
        );
        if (st != TEEGNN_OK) {
            TEE_Free(payload);
            TEE_Free(col_ptr);
            return st;
        }

        const uint8_t *rows = payload;
        const uint8_t *values = rows + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *valid = values + (size_t)enc->header.block_size * sizeof(double);

        for (uint32_t t = 0; t < enc->header.block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * enc->header.block_size + t;
            const uint8_t is_valid = valid[t];
            if (is_valid > 1U) {
                TEE_Free(payload);
                TEE_Free(col_ptr);
                return TEEGNN_ERR_FORMAT;
            }
            if (global_pos >= enc->header.nnz) {
                if (is_valid != 0U) {
                    TEE_Free(payload);
                    TEE_Free(col_ptr);
                    return TEEGNN_ERR_FORMAT;
                }
                continue;
            }
            if (is_valid != 1U) {
                TEE_Free(payload);
                TEE_Free(col_ptr);
                return TEEGNN_ERR_FORMAT;
            }

            while (current_col + 1U <= enc->header.n_nodes &&
                   global_pos >= col_ptr[current_col + 1U]) {
                ++current_col;
                get_row(Y, current_col, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
            }
            if (current_col >= enc->header.n_nodes) {
                TEE_Free(payload);
                TEE_Free(col_ptr);
                return TEEGNN_ERR_FORMAT;
            }

            const uint32_t row = load_u32_slot(rows, t);
            if (row >= enc->header.n_nodes) {
                TEE_Free(payload);
                TEE_Free(col_ptr);
                return TEEGNN_ERR_BOUNDS;
            }
            const double value = load_double_slot(values, t);
            for (uint32_t f = 0; f < cols; ++f) {
                Z[INDEX(row, f)] += value * ctx->temp_row[f];
            }
        }

        TEE_Free(payload);
    }

    TEE_Free(col_ptr);
    return TEEGNN_OK;
}

/* TA入口函数 */
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
    
    /* 分配上下文 */
    ctx = TEE_Malloc(sizeof(gnn_context_t), 0);
    if (ctx == NULL) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    
    memset(ctx, 0, sizeof(gnn_context_t));
    *sess_ctx = (void*)ctx;
    
    DMSG("GNN Nonlinear session opened");
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void* sess_ctx) {
    (void)sess_ctx;
    
    /* 清理上下文 */
    if (ctx != NULL) {
        teegnn_random_engine_wipe(&ctx->rng_data);
        teegnn_random_engine_wipe(&ctx->rng_model);
        free_context_buffers(ctx);
        
        TEE_Free(ctx);
        ctx = NULL;
    }
    
    DMSG("GNN Nonlinear session closed");
}

static TEE_Result init_context(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,  // w1
        TEE_PARAM_TYPE_MEMREF_INPUT,  // seeds and keys
        TEE_PARAM_TYPE_VALUE_INPUT,   // feature_dim, hidden_dim
        TEE_PARAM_TYPE_NONE
    );
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (ctx == NULL) {
        return TEE_ERROR_BAD_STATE;
    }
    free_context_buffers(ctx);
    ctx->initialized = false;

    TEE_Result res;
    size_t feature_dim = params[2].value.a;
    size_t hidden_dim = params[2].value.b;

    char label_data[] = "teegnn-data-owner";
    char label_model[] = "teegnn-model-owner";
    char label_temp[] = "teegnn-temp";
    uint64_t seed_data = 0; 
    uint64_t seed_model = 0; 

    // parse parameters(seeds and keys)
    // format: [seed_data][seed_model][key1][key2]
    uint8_t* secrets = params[1].memref.buffer;
    size_t offset = 0;

    seed_data = *((uint64_t*)(secrets + offset));
    offset += sizeof(uint64_t);
    seed_model = *((uint64_t*)(secrets + offset));
    offset += sizeof(uint64_t);

    ctx->key = TEE_Malloc(2 * sizeof(uint8_t*), 0);
    if (ctx->key == NULL) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < 2; ++i) {
        ctx->key[i] = TEE_Malloc(TEEGNN_AES128_KEY_LEN, 0);
        if (ctx->key[i] == NULL) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        TEE_MemMove(ctx->key[i], secrets + offset,
               TEEGNN_AES128_KEY_LEN);
        offset += TEEGNN_AES128_KEY_LEN;
    }

    // initialize random engine
    if (teegnn_random_engine_init_with_label(&ctx->rng_data,  seed_data, 0, 
                                            label_data, strlen(label_data)) != CSPRNG_OK ||
        teegnn_random_engine_init_with_label(&ctx->rng_model, seed_model, 0, 
                                            label_model, strlen(label_model)) != CSPRNG_OK ||
        teegnn_random_engine_init_with_label(&ctx->rng_temp, 1234, 0, 
                                            label_temp, strlen(label_temp)) != CSPRNG_OK) {
        return TEE_ERROR_BAD_STATE;
    }

    // sdim_mask for w1
    double *w1 = (double*)params[0].memref.buffer;
    res = generate_sdim(&ctx->rng_data, feature_dim, &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    res = generate_sdim(&ctx->rng_temp, hidden_dim, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        return res;
    }

    free_matrix(&ctx->temp_matrix);
    res = init_matrix(&ctx->temp_matrix, feature_dim, hidden_dim);
    if (res != TEE_SUCCESS) {
        return res;
    }

    ctx->row_sum = TEE_Malloc(feature_dim * sizeof(double), 0);
    ctx->col_sum = TEE_Malloc(hidden_dim * sizeof(double), 0);
    if (ctx->row_sum == NULL || ctx->col_sum == NULL) {
        TEE_Free(ctx->row_sum);
        TEE_Free(ctx->col_sum);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    double* row_sum = ctx->row_sum;
    double* col_sum = ctx->col_sum;
    memset(row_sum, 0, feature_dim * sizeof(double));
    memset(col_sum, 0, hidden_dim * sizeof(double));
    
    res = sdim_mask(w1, ctx->temp_matrix.data, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    TEE_MemMove(w1, ctx->temp_matrix.data, feature_dim * hidden_dim * sizeof(double));

    TEE_Free(row_sum);
    TEE_Free(col_sum);
    row_sum = NULL;
    col_sum = NULL;

    ctx->initialized = true;
    
    return TEE_SUCCESS;
}

// message passing and activation function
static TEE_Result secure_compute(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,    // masked matrix
        TEE_PARAM_TYPE_MEMREF_INPUT,    // Blocked CSC
        TEE_PARAM_TYPE_VALUE_INPUT,     // num_nodes, feature_dim
        TEE_PARAM_TYPE_NONE
    );

    TEE_Result res;
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    if (!ctx || !ctx->initialized) {
        return TEE_ERROR_BAD_STATE;
    }

    // parse parameters
    double* y = (double*)params[0].memref.buffer;
    EncryptedBlockedCSC* enc = (EncryptedBlockedCSC *)params[1].memref.buffer;
    uint32_t buffer_size = params[1].memref.size;
    uint32_t rows = params[2].value.a;
    uint32_t cols = params[2].value.b;
    uint32_t layer = enc->header.layer_id;

    // apply sdim unmask
    free_matrix(&ctx->temp_matrix);
    res = init_matrix(&ctx->temp_matrix, rows, cols);
    if (res != TEE_SUCCESS) {
        return res;
    }

    if (layer == 0) {
        res = generate_sdim(&ctx->rng_data, rows, &ctx->temp_SDIM[0]);
        if (res != TEE_SUCCESS) {
            return res;
        }
    }
    
    SDIM* L = &ctx->temp_SDIM[0];
    SDIM* R = &ctx->temp_SDIM[1];
    ctx->row_sum = TEE_Malloc(rows * sizeof(double), 0);
    ctx->col_sum = TEE_Malloc(cols * sizeof(double), 0);
    if (ctx->row_sum == NULL || ctx->col_sum == NULL) {
        TEE_Free(ctx->row_sum);
        TEE_Free(ctx->col_sum);
        return TEE_ERROR_OUT_OF_MEMORY;
    }

    double* row_sum = ctx->row_sum;
    double* col_sum = ctx->col_sum;
    memset(row_sum, 0, rows * sizeof(double));
    memset(col_sum, 0, cols * sizeof(double));
    // 1. calculate the row sum and col sum in order
    ctx->sum = 0;
    ctx->factor = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            row_sum[i] += y[INDEX(i, j)] / L->value[i] * R->h[j];
            col_sum[j] += y[INDEX(i, j)] / L->value[i] * R->value[j];
        }
        ctx->factor += (double)(L->h[i]) / L->value[i];
        ctx->sum += row_sum[i];
    }
    for (size_t j = 0; j < cols; ++j) {
        col_sum[j] /= ctx->factor;
    }
    ctx->sum /= ctx->factor;

    // 2. message passing
    ctx->temp_row = TEE_Malloc(cols * sizeof(double), 0);
    if (ctx->temp_row == NULL) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    blocked_csc_stream_scan(
        enc, 
        ctx->key[layer], 
        TEEGNN_AES128_KEY_LEN, 
        y, 
        cols, 
        ctx->temp_matrix.data
    );

    // 3. activation function
    if (layer == 0) {
        apply_relu(&ctx->temp_matrix);
    } else if (layer == 1) {
        apply_softmax(&ctx->temp_matrix);
    } else {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // return masked matrix to REE
    memset(row_sum, 0, rows * sizeof(double));
    memset(col_sum, 0, cols * sizeof(double));
    res = generate_sdim(&ctx->rng_data, rows, &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    res = generate_sdim(&ctx->rng_model, cols, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    if (layer == 0) {
        sdim_mask(ctx->temp_matrix.data, y, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    } else {
        TEE_MemMove(y, ctx->temp_matrix.data, rows * cols * sizeof(double));
    }
    
    // prepare right sdim for next layer
    res = generate_sdim(&ctx->rng_model, cols, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    
    TEE_Free(row_sum);
    TEE_Free(col_sum);
    row_sum = NULL;
    col_sum = NULL;
    free_matrix(&ctx->temp_matrix);
    
    return TEE_SUCCESS;
}

/* 清理上下文 */
static TEE_Result cleanup_context(uint32_t param_types, TEE_Param params[4]) {
    (void)param_types;
    (void)params;
    
    if (ctx) {
        free_context_buffers(ctx);
        ctx->initialized = false;
    }
    
    return TEE_SUCCESS;
}

/* 获取调试信息 */
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

/* 命令分发 */
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
