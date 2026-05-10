#ifndef TEEGNN_CSPRNG_ADAPTER_C_H
#define TEEGNN_CSPRNG_ADAPTER_C_H

#include <stdint.h>
#include <stddef.h>

#include "csprng.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEEGNN_RANDOM_MATRIX_MIN (-(INT32_C(1) << 15))
#define TEEGNN_RANDOM_MATRIX_MAX (INT32_C(1) << 15)

typedef struct teegnn_random_engine {
    csprng_master_t master;
    csprng_stream_t stream;
} teegnn_random_engine_t;

int teegnn_random_engine_init_with_label(teegnn_random_engine_t *engine,
                                         uint64_t seed,
                                         uint64_t stream_id,
                                         const void *label,
                                         size_t label_len);
int teegnn_random_engine_init(teegnn_random_engine_t *engine,
                              uint64_t seed,
                              uint64_t stream_id,
                              const char *label);
void teegnn_random_engine_wipe(teegnn_random_engine_t *engine);

int teegnn_random_uniform_int(teegnn_random_engine_t *engine,
                              int32_t min_inclusive,
                              int32_t max_exclusive,
                              int32_t *out);
int teegnn_random_uniform_index(teegnn_random_engine_t *engine,
                                uint32_t n,
                                uint32_t *out);
int teegnn_random_nonzero_scale(teegnn_random_engine_t *engine, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TEEGNN_CSPRNG_ADAPTER_C_H */
