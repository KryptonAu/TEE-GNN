#ifndef CSPRNG_H
#define CSPRNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSPRNG_SHA256_SIZE 32U
#define CSPRNG_CHACHA20_KEY_SIZE 32U
#define CSPRNG_CHACHA20_NONCE_SIZE 12U
#define CSPRNG_CHACHA20_BLOCK_SIZE 64U

enum csprng_status {
    CSPRNG_OK = 0,
    CSPRNG_ERR_BAD_PARAMS = -1,
    CSPRNG_ERR_RANGE = -2,
    CSPRNG_ERR_LIMIT = -3
};

typedef struct {
    uint8_t prk[CSPRNG_SHA256_SIZE];
} csprng_master_t;

typedef struct {
    uint8_t key[CSPRNG_CHACHA20_KEY_SIZE];
    uint8_t nonce[CSPRNG_CHACHA20_NONCE_SIZE];
    uint32_t block_counter;
    uint8_t buffer[CSPRNG_CHACHA20_BLOCK_SIZE];
    size_t buffer_pos;
    int exhausted;
} csprng_stream_t;

/*
 * Initializes a deterministic master context from caller-supplied seed bytes.
 * The same seed always produces the same master PRK.
 */
int csprng_master_init(csprng_master_t *master, const void *seed, size_t seed_len);

/*
 * Derives one independent deterministic stream from the master seed.
 * Domain separation binds the fixed label, stream_id and caller label together.
 */
int csprng_stream_init(csprng_stream_t *stream,
                       const csprng_master_t *master,
                       uint64_t stream_id,
                       const void *label,
                       size_t label_len);

/* Deterministic random bytes from the current stream position. */
int csprng_bytes(csprng_stream_t *stream, void *out, size_t out_len);

/* Full-width uniformly random integers. */
int csprng_u32(csprng_stream_t *stream, uint32_t *out);
int csprng_u64(csprng_stream_t *stream, uint64_t *out);

/*
 * Half-open integer ranges [min_inclusive, max_exclusive).
 * Rejection sampling is used so modulo bias is avoided.
 */
int csprng_u32_range(csprng_stream_t *stream,
                     uint32_t min_inclusive,
                     uint32_t max_exclusive,
                     uint32_t *out);
int csprng_u64_range(csprng_stream_t *stream,
                     uint64_t min_inclusive,
                     uint64_t max_exclusive,
                     uint64_t *out);

/* IEEE-754 doubles from deterministic bits. */
int csprng_double(csprng_stream_t *stream, double *out); /* [0, 1) */
int csprng_double_range(csprng_stream_t *stream, double a, double b, double *out); /* [a, b) */
int csprng_double_range_inclusive(csprng_stream_t *stream,
                                  double a,
                                  double b,
                                  double *out); /* [a, b] */

/* Best-effort state wiping for callers that want explicit cleanup. */
void csprng_master_wipe(csprng_master_t *master);
void csprng_stream_wipe(csprng_stream_t *stream);

#ifdef __cplusplus
}
#endif

#endif /* CSPRNG_H */
