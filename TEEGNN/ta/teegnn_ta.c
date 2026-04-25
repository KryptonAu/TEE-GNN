// SPDX-License-Identifier: BSD-2-Clause

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include "teegnn_ta.h"

struct teegnn_context {
    uint32_t node_count;
    uint32_t feature_dim;
    uint32_t hidden_dim;
    uint32_t class_count;
    void *scratch;
    size_t scratch_size;
};

static TEE_Result teegnn_secure_random(void *out, size_t out_len)
{
    /*
     * TODO: Adapter boundary for repository CSPRNG.
     *
     * The Linux simulator links ../CSPRNG/csprng.c directly. A production TA
     * should either port that deterministic CSPRNG into the OP-TEE TA build
     * after auditing libc/API compatibility, or intentionally replace it with
     * TEE_GenerateRandom plus domain-separated derivation. Do not silently use
     * a non-secure PRNG here.
     */
    if (!out && out_len)
        return TEE_ERROR_BAD_PARAMETERS;
    TEE_GenerateRandom(out, out_len);
    return TEE_SUCCESS;
}

static void free_context(struct teegnn_context *ctx)
{
    if (!ctx)
        return;
    if (ctx->scratch)
        TEE_Free(ctx->scratch);
    TEE_Free(ctx);
}

static TEE_Result cmd_init_context(struct teegnn_context *ctx,
                                   uint32_t param_types,
                                   TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);
    uint8_t seed_probe[32];
    TEE_Result res;

    if (!ctx)
        return TEE_ERROR_BAD_STATE;
    if (param_types != expected)
        return TEE_ERROR_BAD_PARAMETERS;

    ctx->node_count = params[0].value.a;
    ctx->feature_dim = params[0].value.b;
    ctx->scratch_size = 4096;
    if (ctx->scratch)
        TEE_Free(ctx->scratch);
    ctx->scratch = TEE_Malloc(ctx->scratch_size, TEE_MALLOC_FILL_ZERO);
    if (!ctx->scratch)
        return TEE_ERROR_OUT_OF_MEMORY;

    res = teegnn_secure_random(seed_probe, sizeof(seed_probe));
    if (res != TEE_SUCCESS)
        return res;

    return TEE_SUCCESS;
}

static TEE_Result cmd_restore_aggregation(struct teegnn_context *ctx,
                                          uint32_t param_types,
                                          TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT);
    if (!ctx)
        return TEE_ERROR_BAD_STATE;
    if (param_types != expected)
        return TEE_ERROR_BAD_PARAMETERS;

    /*
     * TODO:
     * - Read Y1/Y2 masked aggregation shares from params[0]/params[1].
     * - Apply P1^{-1}, P3, P4^{-1}, P6.
     * - Subtract low-rank correction A_hat * M.
     * - Write restored Xbar to params[2].
     */
    (void)params;
    return TEE_SUCCESS;
}

static TEE_Result cmd_apply_activation(struct teegnn_context *ctx,
                                       uint32_t param_types,
                                       TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE);
    if (!ctx)
        return TEE_ERROR_BAD_STATE;
    if (param_types != expected)
        return TEE_ERROR_BAD_PARAMETERS;

    /*
     * TODO:
     * - Apply ReLU in-place for hidden activations.
     * - Generate next low-rank feature mask and scaled permutations.
     * - Return remasking metadata or write protected tensors as needed.
     */
    (void)params;
    return TEE_SUCCESS;
}

static TEE_Result cmd_finalize_result(struct teegnn_context *ctx,
                                      uint32_t param_types,
                                      TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                             TEE_PARAM_TYPE_VALUE_OUTPUT);
    if (!ctx)
        return TEE_ERROR_BAD_STATE;
    if (param_types != expected)
        return TEE_ERROR_BAD_PARAMETERS;

    /*
     * TODO:
     * - Read restored logits and labels.
     * - Compute argmax, prediction buffer, and accuracy/check counters.
     * - Write predictions to params[2] and counters to params[3].
     */
    params[3].value.a = 0;
    params[3].value.b = 0;
    return TEE_SUCCESS;
}

TEE_Result TA_CreateEntryPoint(void)
{
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param params[4],
                                    void **sess_ctx)
{
    struct teegnn_context *ctx = NULL;

    (void)param_types;
    (void)params;
    ctx = TEE_Malloc(sizeof(*ctx), TEE_MALLOC_FILL_ZERO);
    if (!ctx)
        return TEE_ERROR_OUT_OF_MEMORY;

    *sess_ctx = ctx;
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess_ctx)
{
    free_context((struct teegnn_context *)sess_ctx);
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types,
                                      TEE_Param params[4])
{
    (void)sess_ctx;

    switch (cmd_id) {
    case TEEGNN_CMD_INIT_CONTEXT:
        return cmd_init_context((struct teegnn_context *)sess_ctx, param_types, params);
    case TEEGNN_CMD_RESTORE_AGGREGATION:
        return cmd_restore_aggregation((struct teegnn_context *)sess_ctx, param_types, params);
    case TEEGNN_CMD_APPLY_ACTIVATION:
        return cmd_apply_activation((struct teegnn_context *)sess_ctx, param_types, params);
    case TEEGNN_CMD_FINALIZE_RESULT:
        return cmd_finalize_result((struct teegnn_context *)sess_ctx, param_types, params);
    default:
        return TEE_ERROR_NOT_SUPPORTED;
    }
}
