#include "csprng_adapter_c.h"

#include <tee_internal_api.h>

static size_t teegnn_strlen(const char *s)
{
    size_t len = 0U;

    while (s[len] != '\0') {
        ++len;
    }

    return len;
}

static int teegnn_random_init_stream(teegnn_random_engine_t *engine,
                                     uint64_t seed,
                                     uint64_t stream_id,
                                     const void *label,
                                     size_t label_len)
{
    int rc;

    if (engine == NULL || (label == NULL && label_len != 0U)) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    TEE_MemFill(engine, 0, sizeof(*engine));

    rc = csprng_master_init(&engine->master, &seed, sizeof(seed));
    if (rc != CSPRNG_OK) {
        return rc;
    }

    rc = csprng_stream_init(&engine->stream, &engine->master, stream_id,
                            label, label_len);
    if (rc != CSPRNG_OK) {
        csprng_master_wipe(&engine->master);
        TEE_MemFill(engine, 0, sizeof(*engine));
        return rc;
    }

    return CSPRNG_OK;
}

int teegnn_random_engine_init_with_label(teegnn_random_engine_t *engine,
                                         uint64_t seed,
                                         uint64_t stream_id,
                                         const void *label,
                                         size_t label_len)
{
    return teegnn_random_init_stream(engine, seed, stream_id, label, label_len);
}

int teegnn_random_engine_init(teegnn_random_engine_t *engine,
                              uint64_t seed,
                              uint64_t stream_id,
                              const char *label)
{
    if (label == NULL) {
        return teegnn_random_init_stream(engine, seed, stream_id, NULL, 0U);
    }

    return teegnn_random_init_stream(engine, seed, stream_id, label,
                                    teegnn_strlen(label));
}

void teegnn_random_engine_wipe(teegnn_random_engine_t *engine)
{
    if (engine == NULL) {
        return;
    }

    csprng_stream_wipe(&engine->stream);
    csprng_master_wipe(&engine->master);
}

int teegnn_random_uniform(teegnn_random_engine_t *engine,
                          double a,
                          double b,
                          double *out)
{
    if (engine == NULL || out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    return csprng_double_range(&engine->stream, a, b, out);
}

int teegnn_random_uniform_int(teegnn_random_engine_t *engine,
                              int32_t min_inclusive,
                              int32_t max_exclusive,
                              int32_t *out)
{
    uint32_t negative_count;
    uint32_t offset;
    uint32_t span;
    int rc;

    if (engine == NULL || out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }
    if (min_inclusive >= max_exclusive) {
        return CSPRNG_ERR_RANGE;
    }

    if (min_inclusive < 0 && max_exclusive > 0) {
        negative_count = (uint32_t)(-(min_inclusive + INT32_C(1))) + UINT32_C(1);
        span = negative_count + (uint32_t)max_exclusive;
        if (span == 0U) {
            return CSPRNG_ERR_RANGE;
        }
    } else {
        span = (uint32_t)(max_exclusive - min_inclusive);
    }

    rc = csprng_u32_range(&engine->stream, 0U, span, &offset);
    if (rc != CSPRNG_OK) {
        return rc;
    }

    if (min_inclusive < 0 && max_exclusive > 0) {
        negative_count = (uint32_t)(-(min_inclusive + INT32_C(1))) + UINT32_C(1);
        if (offset < negative_count) {
            *out = min_inclusive + (int32_t)offset;
        } else {
            *out = (int32_t)(offset - negative_count);
        }
    } else {
        *out = min_inclusive + (int32_t)offset;
    }

    return CSPRNG_OK;
}

int teegnn_random_uniform_index(teegnn_random_engine_t *engine,
                                uint32_t n,
                                uint32_t *out)
{
    if (engine == NULL || out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }
    if (n == 0U) {
        return CSPRNG_ERR_RANGE;
    }

    return csprng_u32_range(&engine->stream, 0U, n, out);
}

int teegnn_random_matrix_value(teegnn_random_engine_t *engine, int32_t *out)
{
    int32_t value;
    int rc;

    if (out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    rc = teegnn_random_uniform_int(engine,
                                  TEEGNN_RANDOM_MATRIX_MIN,
                                  TEEGNN_RANDOM_MATRIX_MAX,
                                  &value);
    if (rc != CSPRNG_OK) {
        return rc;
    }

    *out = value;
    return CSPRNG_OK;
}

int teegnn_random_nonzero_scale(teegnn_random_engine_t *engine, int32_t *out)
{
    int32_t value;
    int rc;

    if (out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    do {
        rc = teegnn_random_uniform_int(engine,
                                      TEEGNN_RANDOM_MATRIX_MIN,
                                      TEEGNN_RANDOM_MATRIX_MAX,
                                      &value);
        if (rc != CSPRNG_OK) {
            return rc;
        }
    } while (value == 0);

    *out = value;
    return CSPRNG_OK;
}

int teegnn_random_matrix(teegnn_random_engine_t *engine,
                         int rows,
                         int cols,
                         double *row_major_out,
                         size_t out_len)
{
    size_t i;
    size_t total;
    int32_t value;
    int rc;

    if (engine == NULL || row_major_out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }
    if (rows < 0 || cols < 0) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    total = (size_t)rows * (size_t)cols;
    if (rows != 0 && total / (size_t)rows != (size_t)cols) {
        return CSPRNG_ERR_RANGE;
    }
    if (out_len < total) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    for (i = 0U; i < total; ++i) {
        rc = teegnn_random_uniform_int(engine, -256, 255, &value);
        if (rc != CSPRNG_OK) {
            return rc;
        }
        row_major_out[i] = (double)value;
    }

    return CSPRNG_OK;
}
