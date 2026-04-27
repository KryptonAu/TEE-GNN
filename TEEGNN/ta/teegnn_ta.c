// tee_ta_gnn_nonlinear.c
#include <cstddef>
#include <cstdint>
#include <tee_ta_api.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <string.h>
#include <stdlib.h>

#include "teegnn_ta.h"
#include "masks.hpp"

/* 矩阵结构体，用于TEE内部表示 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
    double* data;  // 按列主序存储
} tee_matrix_t;

// Random Permutation Matrix
typedef struct RPM {
    int n;
    int *perm;
    int *value;
} RPM;

// Subtly Designed Invertible Matrix
typedef struct SDIM {
    int n;
    int *perm;
    int *value;
    int *h; // 加性扰动向量，长度为 n
} SDIM;

/* 上下文结构体，存储GNN计算状态 */
typedef struct {
    /* 随机流 */
    csprng_master_t master_;
    csprng_stream_t stream_rpm;
    csprng_stream_t stream_lmm;
    csprng_stream_t stream_sdim;
    csprng_stream_t stream_temp;
    
    /* 参数 */
    size_t num_vertices;
    size_t current_layer;  // 当前层索引
    size_t rank;           // 低秩掩码的秩
    size_t feature_dim;      // 特征维度

    /* 预计算Ahat * u */
    int32_t** lmm_u;
    
    /* 临时缓冲区 */
    tee_matrix_t temp_matrix;
    SDIM temp_SDIM[2];
    RPM n_RPM[4];
    RPM m_RPM[2];
    
    /* 状态标志 */
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
    mat->rows = rows;
    mat->cols = cols;
    mat->data = TEE_Malloc(rows * cols * sizeof(double), 0);
    if (mat->data == NULL) {
        return TEE_ERROR_OUT_OF_MEMORY;
    }
    return TEE_SUCCESS;
}

static void free_matrix(tee_matrix_t* mat) {
    if (mat->data != NULL) {
        TEE_Free(mat->data);
        mat->data = NULL;
    }
}
static void free_rpm(RPM *rpm) {
    if (rpm != 0) {
        TEE_Free(rpm->perm);
        TEE_Free(rpm->value);
        rpm->perm = NULL;
        rpm->value = NULL;
    }
}
static void free_sdim(SDIM *sdim) {
    if (sdim != 0) {
        TEE_Free(sdim->perm);
        TEE_Free(sdim->value);
        TEE_Free(sdim->h);
        sdim->perm = NULL;
        sdim->value = NULL;
        sdim->h = NULL;
    }
}
static void free_lmm(LMM *lmm) {
    if (lmm != 0) {
        TEE_Free(lmm->u);
        TEE_Free(lmm->v);
        lmm->u = NULL;
        lmm->v = NULL;
    }
}

// matrices are in column-major order.
static void sdim_mask(double *in, double *out, SDIM *L, SDIM *R) {
    int r = L->n;
    int c = R->n;
    double factor = 1.0;
    double sum = 0.0;
    double* row_sum = static_cast<double *>(std::calloc(static_cast<size_t>(r), sizeof(double)));
    double* col_sum = static_cast<double *>(std::calloc(static_cast<size_t>(c), sizeof(double)));
    for (int j = 0; j < c; ++j) {
        factor += static_cast<double>(R->h[j]) / R->value[j];
        for (int i = 0; i < r; ++i) {
            row_sum[i] += in[i + R->perm[j] * r] * R->h[j] / R->value[j];
            col_sum[j] += in[i + j * r];
        }
    }
    for (int j = 0; j < c; ++j) {
        sum += col_sum[R->perm[j]] * R->h[j] / R->value[j];
    }
    for (int j = 0; j < c; ++j) {
        int n_j = R->perm[j];
        for (int i = 0; i < r; ++i) {
            int n_i = L->perm[i];
            out[i + j * r] = in[n_i + n_j * r] * L->value[i] / R->value[j] + 
                             col_sum[n_j] * L->h[i] / R->value[j] -
                             row_sum[n_i] * L->value[i] / R->value[j] / factor -
                             sum * L->h[i] / R->value[j] / factor;
        }
    }
    free(row_sum);
    free(col_sum);
}

static void sdim_unmask(double *in, double *out, SDIM *L, SDIM *R) {
    int r = L->n;
    int c = R->n;
    double factor = 1.0;
    double sum = 0.0;
    double* row_sum = static_cast<double *>(std::calloc(static_cast<size_t>(r), sizeof(double)));
    double* col_sum = static_cast<double *>(std::calloc(static_cast<size_t>(c), sizeof(double)));
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            row_sum[i] += in[i + j * r] * R->h[j] / L->value[i];
            col_sum[j] += in[i + j * r] * R->value[j] / L->value[i];
        }
    }
    for (int i = 0; i < r; ++i) {
        factor += static_cast<double>(L->h[i]) / L->value[i];
        sum += row_sum[i];
    }
    for (int j = 0; j < c; ++j) {
        int n_j = R->perm[j];
        for (int i = 0; i < r; ++i) {
            int n_i = L->perm[i];
            out[n_i + n_j * r] = in[i + j * r] * R->value[j] / L->value[i] + 
                                 row_sum[i] -
                                 col_sum[j] * L->h[i] / L->value[i] / factor -
                                 sum * L->h[i] / L->value[i] / factor;
        }
    }
    free(row_sum);
    free(col_sum);
}

static void apply_relu(tee_matrix_t* mat) {
    uint32_t total = mat->rows * mat->cols;
    for (uint32_t i = 0; i < total; i++) {
        mat->data[i] = mat->data[i] > 0 ? mat->data[i] : 0;
    }
}

static void apply_softmax(tee_matrix_t* mat) {
    for (uint32_t i = 0; i < mat->rows; i++) {
        // 找最大值（数值稳定）
        double max_val = mat->data[i];
        for (uint32_t j = 1; j < mat->cols; j++) {
            double val = mat->data[i + j * mat->rows];
            if (val > max_val) max_val = val;
        }
        
        // 计算exp和sum
        double sum = 0;
        for (uint32_t j = 0; j < mat->cols; j++) {
            double val = mat->data[i + j * mat->rows];
            mat->data[i + j * mat->rows] = Exp(val - max_val);
            sum += mat->data[i + j * mat->rows];
        }
        
        // 归一化
        for (uint32_t j = 0; j < mat->cols; j++) {
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
        csprng_master_wipe(&master_);
        csprng_stream_wipe(&stream_rpm);
        csprng_stream_wipe(&stream_sdim);
        csprng_stream_wipe(&stream_lmm);
        csprng_stream_wipe(&stream_temp);
        if (ctx->LMM_u != NULL) {
            for (uint32_t i = 0; i < 2; i++) { 
                if (ctx->LMM_u[i] != NULL) {
                    TEE_Free(ctx->LMM_u[i]);
                }
            }
            TEE_Free(ctx->LMM_u);
        }
        
        for (int i = 0; i < 2; ++i) {
            free_sdim(&ctx->temp_SDIM);
        }
        for (int i = 0; i < 4; ++i) {
            free_rpm(&ctx->n_RPM);
        }
        for (int i = 0; i < 2; ++i) {
            free_rpm(&ctx->m_RPM);
        }
        
        free_matrix(&ctx->temp_matrix);
        
        TEE_Free(ctx);
        ctx = NULL;
    }
    
    DMSG("GNN Nonlinear session closed");
}

/* 初始化上下文 */
static TEE_Result init_context(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INPUT,  // low_rank_mask precomp
        TEE_PARAM_TYPE_VALUE_INPUT,   // num_vertices, rank
        TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE
    );
    
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    if (ctx == NULL) {
        return TEE_ERROR_BAD_STATE;
    }

    char label_rpm = "";
    char label_sdim = "";
    char label_lmm = "";
    char label_temp = "";
    uint64_t seed = 114514; 

    csprng_master_init(&ctx->master_, &seed, sizeof(seed));
    csprng_stream_init(&ctx->stream_rpm,  &ctx->master_, 0, label_rpm, strlen(label_rpm));
    csprng_stream_init(&ctx->stream_sdim, &ctx->master_, 1, label_sdim, strlen(label_sdim));
    csprng_stream_init(&ctx->stream_lmm,  &ctx->master_, 2, label_lmm, strlen(label_lmm));
    csprng_stream_init(&ctx->stream_rpm,  &ctx->master_, 3, label_temp, strlen(label_temp));

    // 解析参数
    ctx->num_vertices = params[1].value.a;
    ctx->rank = params[1].value.b;
    
    ctx->lmm_u = TEE_Malloc(2 * sizeof(int32_t*), 0);

    size_t lmm_len = ctx->rank * ctx->num_vertices;
    size_t offset = 0;
    for (int i = 0; i < 2; i++) {
        ctx->lmm_u[i] = TEE_Malloc(lmm_len * sizeof(int32_t), 0);
        if (ctx->lmm_u[i] == NULL) {
            return TEE_ERROR_OUT_OF_MEMORY;
        }
        memcpy(ctx->lmm_u[i], params[0].memref.buffer + offset, lmm_len * sizeof(int32_t));
        offset += lmm_len * sizeof(int32_t);
    }
    
    ctx->initialized = true;
    
    return TEE_SUCCESS;
}

static TEE_Result rpm_mask(double *in, double *out, RPM *L, RPM *R) {
    int r = L->n;
    int c = R->n;
    for (int j = 0; j < c; ++j) {
        int new_j = R->perm[j];
        for (int i = 0; i < r; ++i) {
            int new_i = L->perm[i];
            int mask_value = 0;
            out[i + j * r] = in[new_i + new_j * r] * L->value[i] / R->value[j];
        }
    }
    return TEE_SUCCESS;
}

static TEE_Result rpm_unmask(double *in, double *out, RPM *L, RPM *R, bool add_tag) {
    int r = L->n;
    int c = R->n;
    for (int j = 0; j < c; ++j) {
        int new_j = R->perm[j];
        for (int i = 0; i < r; ++i) {
            int new_i = L->perm[i];
            if (add_tag) {
                out[new_i + new_j * r] += in[i + j * r] / L->value[i] * R->value[j];
            } else {
                out[new_i + new_j * r] = in[i + j * r] / L->value[i] * R->value[j];
            }
        }
    }
    return TEE_SUCCESS;
}

/* 还原邻居聚合结果 */
static TEE_Result restore_aggregation(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,   // 矩阵1
        TEE_PARAM_TYPE_MEMREF_INPUT,   // 矩阵2
        TEE_PARAM_TYPE_VALUE_INPUT,
        TEE_PARAM_TYPE_NONE
    );
    
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    if (!ctx || !ctx->initialized) {
        return TEE_ERROR_BAD_STATE;
    }

    // 从参数中提取附加信息
    uint32_t layer_idx = params[2].value.a;
    uint32_t feature_dim = params[2].value.b;

    // 重置临时矩阵
    free_matrix(&ctx->temp_matrix);
    init_matrix(&ctx->temp_matrix, ctx->num_vertices, feature_dim);
    
    // 1. 解密两个输入矩阵
    double* enc1 = (double*)params[0].memref.buffer;
    double* enc2 = (double*)params[1].memref.buffer;

    rpm_unmask(enc1, ctx->temp_matrix.data, &ctx->n_RPM[0], &ctx->m_RPM[0], false);
    rpm_unmask(enc2, ctx->temp_matrix.data, &ctx->n_RPM[1], &ctx->m_RPM[1], true);

    for (size_t j = 0; j < feature_dim; j++) {
        for (size_t i = 0; i < ctx->num_vertices; i++) {
            int32_t masked_val = 0;
            for (size_t k = 0; k < ctx->rank; k++) {
                masked_val += ctx->lmm_u[layer_idx][i + k * ctx->num_vertices];
            }
            ctx->temp_matrix.data[i + j * ctx->num_vertices] -= masked_val;
                    ;
        }
    }
    
    sdim_mask(ctx->temp_matrix.data, enc1, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);
    
    DMSG("Nonlinear layer %u completed (activation=%u)",
         layer_idx, activation);
    
    return TEE_SUCCESS;
}

/* 非线性层计算 */
static TEE_Result nonlinear_layer(uint32_t param_types, TEE_Param params[4]) {
    uint32_t exp_param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INOUT,
        TEE_PARAM_TYPE_MEMREF_OUTPUT,
        TEE_PARAM_TYPE_VALUE_INPUT,
        TEE_PARAM_TYPE_NONE
    );
    
    if (param_types != exp_param_types) {
        return TEE_ERROR_BAD_PARAMETERS;
    }
    
    if (!ctx || !ctx->initialized) {
        return TEE_ERROR_BAD_STATE;
    }
    
    // 从参数中提取附加信息
    uint32_t layer_idx = (params[2].value.a >> 16) & 0xFFFF;
    uint32_t activation = params[2].value.a & 0xFFFF;
    uint32_t feature_dim = params[2].value.b;

    free_matrix(&ctx->temp_matrix);
    init_matrix(&ctx->temp_matrix, ctx->num_vertices, feature_dim);
    
    double* linear = (double*)params[0].memref.buffer;
    
    sdim_unmask(linear, ctx->temp_matrix.data, &ctx->temp_SDIM[0], &ctx->temp_SDIM[1]);

    // activation
    if (activation == 0) {  // ReLU
        apply_relu(&ctx->temp_matrix);
    } else if (activation == 1) {  // Softmax
        apply_softmax(&ctx->temp_matrix);
    } else {
        return TEE_ERROR_GNN_INVALID_INPUT;
    }
    
    if (activation == 1) {  // Softmax
        memcpy(enc1, ctx->temp_matrix.data,
               ctx->num_vertices * feature_dim * sizeof(double));
        memcpy(enc2, ctx->temp_matrix.data,
               ctx->num_vertices * feature_dim * sizeof(double));
    } else {  // ReLU
        encrypt_matrix(&ctx->temp_matrix, enc1, 1, 6 + 4 * layer_idx);
        encrypt_matrix(&ctx->temp_matrix, enc2, 3, 8 + 4 * layer_idx);
    }
    
    DMSG("Nonlinear layer %u completed (activation=%u)",
         layer_idx, activation);
    
    return TEE_SUCCESS;
}

/* 清理上下文 */
static TEE_Result cleanup_context(uint32_t param_types, TEE_Param params[4]) {
    (void)param_types;
    (void)params;
    
    if (ctx) {
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
    
    int written = snprintf(buf, buf_size,
                          "GNN TA Status:\n"
                          "Initialized: %s\n"
                          "Vertices: %u\n"
                          "Temp matrices: %ux%u",
                          ctx->initialized ? "Yes" : "No",
                          ctx->num_vertices,
                          ctx->temp_matrix.rows,
                          ctx->temp_matrix.cols);
    
    if (written < 0 || (size_t)written >= buf_size) {
        return TEE_ERROR_SHORT_BUFFER;
    }
    
    return TEE_SUCCESS;
}

/* 命令分发 */
TEE_Result TA_InvokeCommandEntryPoint(void* sess_ctx, uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4]) {
    (void)sess_ctx;
    
    switch (cmd_id) {
        case TA_CMD_INIT_CONTEXT:
            return init_context(param_types, params);
        case TA_CMD_RESTORE:
            return restore_aggregation(param_types, params);
        case TA_CMD_NONLINEAR_LAYER:
            return nonlinear_layer(param_types, params);
        case TA_CMD_CLEANUP_CONTEXT:
            return cleanup_context(param_types, params);
        case TA_CMD_GET_DEBUG_INFO:
            return get_debug_info(param_types, params);
        default:
            return TEE_ERROR_NOT_SUPPORTED;
    }
}
