#ifndef TEEGNN_CRYPTO_H
#define TEEGNN_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#include "teegnn_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEEGNN_GCM_NONCE_LEN 12
#define TEEGNN_GCM_TAG_LEN 16
#define TEEGNN_AES128_KEY_LEN 16
#define TEEGNN_AES256_KEY_LEN 32

teegnn_status_t teegnn_aes_gcm_encrypt(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *plaintext,
    size_t plaintext_len,
    uint8_t *ciphertext,
    uint8_t tag[TEEGNN_GCM_TAG_LEN]
);

teegnn_status_t teegnn_aes_gcm_decrypt(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *ciphertext,
    size_t ciphertext_len,
    const uint8_t tag[TEEGNN_GCM_TAG_LEN],
    uint8_t *plaintext
);

#ifdef __cplusplus
}
#endif

#endif
