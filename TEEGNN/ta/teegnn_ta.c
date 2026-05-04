// tee_ta_gnn_nonlinear.c
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "teegnn_ta.h"
#include "csprng_adapter_c.h"

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <tee_ta_api.h> 

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

/* 上下文结构体，存储GNN计算状态 */
typedef struct {
    /* 随机流 */
    teegnn_random_engine_t rng_spm;
    teegnn_random_engine_t rng_lmm;
    teegnn_random_engine_t rng_sdim;
    teegnn_random_engine_t rng_temp;
    
    /* 参数 */
    size_t num_vertices;

    /* 临时缓冲区 */
    tee_matrix_t temp_matrix;
    // temp_SDIM[0].h and temp_SDIM[1].h will be used as temporary buffers
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

// matrices are in column-major order.
static TEE_Result sdim_mask(double *in, double *out, SDIM *L, SDIM *R) {
    size_t r = L->n;
    size_t c = R->n;
    double factor = 1.0;
    double sum = 0.0;
    double* row_sum = TEE_Malloc(r * sizeof(double), 0);
    double* col_sum = TEE_Malloc(c * sizeof(double), 0);
    if (row_sum == NULL || col_sum == NULL) {
        TEE_Free(row_sum);
        TEE_Free(col_sum);
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    memset(row_sum, 0, r * sizeof(double));
    memset(col_sum, 0, c * sizeof(double));

    for (size_t j = 0; j < c; ++j) {
        factor += (double)(R->h[j]) / R->value[j];
        for (size_t i = 0; i < r; ++i) {
            row_sum[i] += in[i + R->perm[j] * r] / R->value[j] * R->h[j];
            col_sum[j] += in[i + j * r];
        }
    }
    for (size_t j = 0; j < c; ++j) {
        sum += col_sum[R->perm[j]] * R->h[j] / R->value[j];
    }
    for (size_t j = 0; j < c; ++j) {
        size_t n_j = R->perm[j];
        for (size_t i = 0; i < r; ++i) {
            size_t n_i = L->perm[i];
            out[i + j * r] = in[n_i + n_j * r] * L->value[i] / R->value[j] + 
                             col_sum[n_j] * L->h[i] / R->value[j] -
                             row_sum[n_i] * L->value[i] / R->value[j] / factor -
                             sum * L->h[i] / R->value[j] / factor;
        }
    }
    TEE_Free(row_sum);
    TEE_Free(col_sum);

    return TEE_SUCCESS;
}

static TEE_Result sdim_unmask(double *in, double *out, SDIM *L, SDIM *R) {
    size_t r = L->n;
    size_t c = R->n;
    double factor = 1.0;
    double sum = 0.0;
    double* row_sum = TEE_Malloc(r * sizeof(double), 0);
    double* col_sum = TEE_Malloc(c * sizeof(double), 0);
    if (row_sum == NULL || col_sum == NULL) {
        TEE_Free(row_sum);
        TEE_Free(col_sum);
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    memset(row_sum, 0, r * sizeof(double));
    memset(col_sum, 0, c * sizeof(double));

    for (size_t j = 0; j < c; ++j) {
        for (size_t i = 0; i < r; ++i) {
            row_sum[i] += in[i + j * r] / L->value[i] * R->h[j];
            col_sum[j] += in[i + j * r] * R->value[j] / L->value[i];
        }
    }
    for (size_t i = 0; i < r; ++i) {
        factor += (double)(L->h[i]) / L->value[i];
        sum += row_sum[i];
    }
    for (size_t j = 0; j < c; ++j) {
        int n_j = R->perm[j];
        for (size_t i = 0; i < r; ++i) {
            int n_i = L->perm[i];
            out[n_i + n_j * r] = in[i + j * r] / L->value[i] * R->value[j] + 
                                 row_sum[i] -
                                 col_sum[j] / L->value[i] * L->h[i] / factor -
                                 sum / L->value[i] * L->h[i] / factor;
        }
    }
    TEE_Free(row_sum);
    TEE_Free(col_sum);

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
        teegnn_random_engine_wipe(&ctx->rng_spm);
        teegnn_random_engine_wipe(&ctx->rng_sdim);
        teegnn_random_engine_wipe(&ctx->rng_lmm);
        teegnn_random_engine_wipe(&ctx->rng_temp);
        free_context_buffers(ctx);
        
        TEE_Free(ctx);
        ctx = NULL;
    }
    
    DMSG("GNN Nonlinear session closed");
}

/* 初始化上下文 */
static TEE_Result init_context(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,  // w1
        TEE_PARAM_TYPE_MEMREF_INPUT,  // low_rank_mask precompute
        TEE_PARAM_TYPE_VALUE_INPUT,   // num_vertices, rank
        TEE_PARAM_TYPE_VALUE_INPUT    // feature_dim, hidden_dim
    );
    TEE_Result res;
    const uint8_t *lmm_buffer;
    size_t lmm_len;
    size_t lmm_bytes;
    size_t expected_bytes;
    size_t offset = 0;
    size_t feature_dim;
    size_t hidden_dim;
    
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    if (ctx == NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    free_context_buffers(ctx);
    ctx->initialized = false;

    char label_spm[] = "teegnn-spm";
    char label_sdim[] = "teegnn-sdim";
    uint64_t seed = 114514; 

    if (teegnn_random_engine_init_with_label(&ctx->rng_spm,  seed, 0, label_spm, strlen(label_spm)) != CSPRNG_OK ||
        teegnn_random_engine_init_with_label(&ctx->rng_sdim, seed, 2, label_sdim, strlen(label_sdim)) != CSPRNG_OK) {
        return TEE_ERROR_BAD_STATE;
    }

    ctx->num_vertices = params[2].value.a;
    feature_dim = params[3].value.a;
    hidden_dim = params[3].value.b;
    if (ctx->num_vertices == 0) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    // sdim_mask for w1
    double *w1 = (double*)params[0].memref.buffer;
    res = generate_sdim(&ctx->rng_temp, hidden_dim, &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    res = generate_sdim(&ctx->rng_sdim, feature_dim, &ctx->temp_SDIM[1]);
    if (res != TEE_SUCCESS) {
        return res;
    }

    free_matrix(&ctx->temp_matrix);
    res = init_matrix(&ctx->temp_matrix, feature_dim, hidden_dim);
    if (res != TEE_SUCCESS) {
        return res;
    }
    
    res = sdim_mask(w1, ctx->temp_matrix.data, &ctx->temp_SDIM[1], &ctx->temp_SDIM[0]);
    if (res != TEE_SUCCESS) {
        return res;
    }
    memcpy(w1, ctx->temp_matrix.data, feature_dim * hidden_dim * sizeof(double));

    ctx->initialized = true;
    
    return TEE_SUCCESS;
}

// message passing and activation function
static TEE_Result secure_compute(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,    // masked matrix
        TEE_PARAM_TYPE_MEMREF_INPUT,    // Blocked CSC
        TEE_PARAM_TYPE_VALUE_INPUT,
        TEE_PARAM_TYPE_NONE
    );
    
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
    
    memcpy(buf, ctx->temp_SDIM[1].h, buf_size);
    
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
