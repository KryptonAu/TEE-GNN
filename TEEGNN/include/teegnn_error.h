#ifndef TEEGNN_ERROR_H
#define TEEGNN_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TEEGNN_OK = 0,
    TEEGNN_ERR_INVALID_ARG = -1,
    TEEGNN_ERR_ALLOC = -2,
    TEEGNN_ERR_CRYPTO = -3,
    TEEGNN_ERR_AUTH_FAIL = -4,
    TEEGNN_ERR_FORMAT = -5,
    TEEGNN_ERR_BOUNDS = -6,
    TEEGNN_ERR_PERMUTATION = -7
} teegnn_status_t;

#ifdef __cplusplus
}
#endif

#endif
