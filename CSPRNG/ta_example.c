/*
 * OP-TEE TA integration example.
 *
 * This file shows how to use the deterministic CSPRNG core inside a standard
 * Trusted Application. It is not part of the default Linux build because this
 * machine does not have an OP-TEE TA toolchain installed.
 */

#ifdef CSPRNG_WITH_OPTEE_TA

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include "csprng.h"

#define TA_CSPRNG_CMD_SET_MASTER_SEED   0U
#define TA_CSPRNG_CMD_STREAM_BYTES      1U
#define TA_CSPRNG_CMD_U64_RANGE         2U
#define TA_CSPRNG_CMD_DOUBLE_RANGE      3U

struct ta_csprng_u64_range_req {
    uint64_t min_inclusive;
    uint64_t max_exclusive;
};

struct ta_csprng_double_range_req {
    double a;
    double b;
    uint32_t inclusive_upper;
};

static csprng_master_t g_master;
static bool g_master_set;

static uint64_t value_to_u64(const TEE_Param *param)
{
    return ((uint64_t)param->value.b << 32) | param->value.a;
}

static TEE_Result map_rc(int rc)
{
    switch (rc) {
    case CSPRNG_OK:
        return TEE_SUCCESS;
    case CSPRNG_ERR_BAD_PARAMS:
    case CSPRNG_ERR_RANGE:
        return TEE_ERROR_BAD_PARAMETERS;
    case CSPRNG_ERR_LIMIT:
        return TEE_ERROR_OVERFLOW;
    default:
        return TEE_ERROR_GENERIC;
    }
}

static TEE_Result ta_set_master_seed(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                              TEE_PARAM_TYPE_NONE,
                                              TEE_PARAM_TYPE_NONE,
                                              TEE_PARAM_TYPE_NONE);

    if (param_types != expected) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (params[0].memref.buffer == NULL || params[0].memref.size == 0U) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    csprng_master_wipe(&g_master);
    g_master_set = false;

    if (csprng_master_init(&g_master, params[0].memref.buffer, params[0].memref.size) != CSPRNG_OK) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    g_master_set = true;
    return TEE_SUCCESS;
}

static TEE_Result ta_stream_bytes(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                              TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                              TEE_PARAM_TYPE_NONE,
                                              TEE_PARAM_TYPE_NONE);
    csprng_stream_t stream;
    uint64_t stream_id;
    int rc;

    if (param_types != expected || !g_master_set) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    stream_id = value_to_u64(&params[0]);
    rc = csprng_stream_init(&stream, &g_master, stream_id, "ta-bytes", 8U);
    if (rc != CSPRNG_OK) {
        return map_rc(rc);
    }

    rc = csprng_bytes(&stream, params[1].memref.buffer, params[1].memref.size);
    csprng_stream_wipe(&stream);
    return map_rc(rc);
}

static TEE_Result ta_u64_range(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                              TEE_PARAM_TYPE_MEMREF_INPUT,
                                              TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                              TEE_PARAM_TYPE_NONE);
    const struct ta_csprng_u64_range_req *req;
    csprng_stream_t stream;
    uint64_t out_value;
    int rc;

    if (param_types != expected || !g_master_set) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (params[1].memref.buffer == NULL ||
        params[1].memref.size != sizeof(struct ta_csprng_u64_range_req) ||
        params[2].memref.buffer == NULL ||
        params[2].memref.size < sizeof(uint64_t)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const struct ta_csprng_u64_range_req *)params[1].memref.buffer;

    rc = csprng_stream_init(&stream, &g_master, value_to_u64(&params[0]), "ta-u64", 6U);
    if (rc != CSPRNG_OK) {
        return map_rc(rc);
    }

    rc = csprng_u64_range(&stream, req->min_inclusive, req->max_exclusive, &out_value);
    if (rc == CSPRNG_OK) {
        TEE_MemMove(params[2].memref.buffer, &out_value, sizeof(out_value));
        params[2].memref.size = sizeof(out_value);
    }

    csprng_stream_wipe(&stream);
    return map_rc(rc);
}

static TEE_Result ta_double_range(uint32_t param_types, TEE_Param params[4])
{
    const uint32_t expected = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INPUT,
                                              TEE_PARAM_TYPE_MEMREF_INPUT,
                                              TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                              TEE_PARAM_TYPE_NONE);
    const struct ta_csprng_double_range_req *req;
    csprng_stream_t stream;
    double out_value;
    int rc;

    if (param_types != expected || !g_master_set) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    if (params[1].memref.buffer == NULL ||
        params[1].memref.size != sizeof(struct ta_csprng_double_range_req) ||
        params[2].memref.buffer == NULL ||
        params[2].memref.size < sizeof(double)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    req = (const struct ta_csprng_double_range_req *)params[1].memref.buffer;

    rc = csprng_stream_init(&stream, &g_master, value_to_u64(&params[0]), "ta-double", 9U);
    if (rc != CSPRNG_OK) {
        return map_rc(rc);
    }

    if (req->inclusive_upper != 0U) {
        rc = csprng_double_range_inclusive(&stream, req->a, req->b, &out_value);
    } else {
        rc = csprng_double_range(&stream, req->a, req->b, &out_value);
    }

    if (rc == CSPRNG_OK) {
        TEE_MemMove(params[2].memref.buffer, &out_value, sizeof(out_value));
        params[2].memref.size = sizeof(out_value);
    }

    csprng_stream_wipe(&stream);
    return map_rc(rc);
}

TEE_Result TA_CreateEntryPoint(void)
{
    g_master_set = false;
    csprng_master_wipe(&g_master);
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    csprng_master_wipe(&g_master);
    g_master_set = false;
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param params[4],
                                    void **sess_ctx)
{
    (void)param_types;
    (void)params;
    (void)sess_ctx;
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess_ctx)
{
    (void)sess_ctx;
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types,
                                      TEE_Param params[4])
{
    (void)sess_ctx;

    switch (cmd_id) {
    case TA_CSPRNG_CMD_SET_MASTER_SEED:
        return ta_set_master_seed(param_types, params);
    case TA_CSPRNG_CMD_STREAM_BYTES:
        return ta_stream_bytes(param_types, params);
    case TA_CSPRNG_CMD_U64_RANGE:
        return ta_u64_range(param_types, params);
    case TA_CSPRNG_CMD_DOUBLE_RANGE:
        return ta_double_range(param_types, params);
    default:
        return TEE_ERROR_NOT_SUPPORTED;
    }
}

/*
 * Optional master-seed source example:
 *
 * A TA may call TEE_GenerateRandom() once to obtain a fresh master seed and
 * then keep that seed in secure storage. The deterministic CSPRNG itself still
 * remains seed-driven and reproducible once the stored seed is fixed.
 */

#else

int ta_example_c_present_for_reference_only(void)
{
    return 0;
}

#endif /* CSPRNG_WITH_OPTEE_TA */
