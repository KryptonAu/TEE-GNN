#include "crypto.h"

#include <limits.h>
#include <openssl/evp.h>

static const EVP_CIPHER *select_aes_gcm_cipher(size_t key_len) {
    if (key_len == TEEGNN_AES128_KEY_LEN) {
        return EVP_aes_128_gcm();
    }
    if (key_len == TEEGNN_AES256_KEY_LEN) {
        return EVP_aes_256_gcm();
    }
    return NULL;
}

static teegnn_status_t validate_common_args(
    const uint8_t *key,
    size_t key_len,
    const uint8_t nonce[TEEGNN_GCM_NONCE_LEN],
    const uint8_t *aad,
    size_t aad_len,
    const uint8_t *input,
    size_t input_len,
    const uint8_t tag[TEEGNN_GCM_TAG_LEN],
    uint8_t *output
) {
    if (key == NULL || nonce == NULL || tag == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (select_aes_gcm_cipher(key_len) == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (aad_len > 0 && aad == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (input_len > 0 && (input == NULL || output == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    return TEEGNN_OK;
}

static teegnn_status_t encrypt_update_bytes(
    EVP_CIPHER_CTX *ctx,
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *produced_total
) {
    size_t in_off = 0;
    size_t out_off = 0;
    while (in_off < input_len) {
        const size_t remaining = input_len - in_off;
        const int chunk = remaining > INT_MAX ? INT_MAX : (int)remaining;
        int produced = 0;
        if (EVP_EncryptUpdate(ctx, output + out_off, &produced, input + in_off, chunk) != 1) {
            return TEEGNN_ERR_CRYPTO;
        }
        if (produced < 0) {
            return TEEGNN_ERR_CRYPTO;
        }
        in_off += (size_t)chunk;
        out_off += (size_t)produced;
    }
    *produced_total = out_off;
    return TEEGNN_OK;
}

static teegnn_status_t decrypt_update_bytes(
    EVP_CIPHER_CTX *ctx,
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t *produced_total
) {
    size_t in_off = 0;
    size_t out_off = 0;
    while (in_off < input_len) {
        const size_t remaining = input_len - in_off;
        const int chunk = remaining > INT_MAX ? INT_MAX : (int)remaining;
        int produced = 0;
        if (EVP_DecryptUpdate(ctx, output + out_off, &produced, input + in_off, chunk) != 1) {
            return TEEGNN_ERR_CRYPTO;
        }
        if (produced < 0) {
            return TEEGNN_ERR_CRYPTO;
        }
        in_off += (size_t)chunk;
        out_off += (size_t)produced;
    }
    *produced_total = out_off;
    return TEEGNN_OK;
}

static teegnn_status_t encrypt_update_aad(
    EVP_CIPHER_CTX *ctx,
    const uint8_t *aad,
    size_t aad_len
) {
    size_t off = 0;
    while (off < aad_len) {
        const size_t remaining = aad_len - off;
        const int chunk = remaining > INT_MAX ? INT_MAX : (int)remaining;
        int ignored = 0;
        if (EVP_EncryptUpdate(ctx, NULL, &ignored, aad + off, chunk) != 1) {
            return TEEGNN_ERR_CRYPTO;
        }
        off += (size_t)chunk;
    }
    return TEEGNN_OK;
}

static teegnn_status_t decrypt_update_aad(
    EVP_CIPHER_CTX *ctx,
    const uint8_t *aad,
    size_t aad_len
) {
    size_t off = 0;
    while (off < aad_len) {
        const size_t remaining = aad_len - off;
        const int chunk = remaining > INT_MAX ? INT_MAX : (int)remaining;
        int ignored = 0;
        if (EVP_DecryptUpdate(ctx, NULL, &ignored, aad + off, chunk) != 1) {
            return TEEGNN_ERR_CRYPTO;
        }
        off += (size_t)chunk;
    }
    return TEEGNN_OK;
}

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
) {
    teegnn_status_t st = validate_common_args(
        key, key_len, nonce, aad, aad_len, plaintext, plaintext_len, tag, ciphertext
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    const EVP_CIPHER *cipher = select_aes_gcm_cipher(key_len);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        return TEEGNN_ERR_ALLOC;
    }

    if (EVP_EncryptInit_ex(ctx, cipher, NULL, NULL, NULL) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, TEEGNN_GCM_NONCE_LEN, NULL) != 1 ||
        EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return TEEGNN_ERR_CRYPTO;
    }

    st = encrypt_update_aad(ctx, aad, aad_len);
    if (st == TEEGNN_OK) {
        size_t produced = 0;
        st = encrypt_update_bytes(ctx, plaintext, plaintext_len, ciphertext, &produced);
        if (st == TEEGNN_OK && produced != plaintext_len) {
            st = TEEGNN_ERR_CRYPTO;
        }
    }

    if (st == TEEGNN_OK) {
        int final_len = 0;
        uint8_t final_buf[EVP_MAX_BLOCK_LENGTH];
        uint8_t *final_out = ciphertext == NULL ? final_buf : ciphertext + plaintext_len;
        if (EVP_EncryptFinal_ex(ctx, final_out, &final_len) != 1 ||
            final_len != 0 ||
            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, TEEGNN_GCM_TAG_LEN, tag) != 1) {
            st = TEEGNN_ERR_CRYPTO;
        }
    }

    EVP_CIPHER_CTX_free(ctx);
    return st;
}

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
) {
    teegnn_status_t st = validate_common_args(
        key, key_len, nonce, aad, aad_len, ciphertext, ciphertext_len, tag, plaintext
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    const EVP_CIPHER *cipher = select_aes_gcm_cipher(key_len);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        return TEEGNN_ERR_ALLOC;
    }

    if (EVP_DecryptInit_ex(ctx, cipher, NULL, NULL, NULL) != 1 ||
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, TEEGNN_GCM_NONCE_LEN, NULL) != 1 ||
        EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return TEEGNN_ERR_CRYPTO;
    }

    st = decrypt_update_aad(ctx, aad, aad_len);
    if (st == TEEGNN_OK) {
        size_t produced = 0;
        st = decrypt_update_bytes(ctx, ciphertext, ciphertext_len, plaintext, &produced);
        if (st == TEEGNN_OK && produced != ciphertext_len) {
            st = TEEGNN_ERR_CRYPTO;
        }
    }

    if (st == TEEGNN_OK &&
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, TEEGNN_GCM_TAG_LEN, (void *)tag) != 1) {
        st = TEEGNN_ERR_CRYPTO;
    }

    if (st == TEEGNN_OK) {
        int final_len = 0;
        uint8_t final_buf[EVP_MAX_BLOCK_LENGTH];
        uint8_t *final_out = plaintext == NULL ? final_buf : plaintext + ciphertext_len;
        const int final_ok = EVP_DecryptFinal_ex(
            ctx,
            final_out,
            &final_len
        );
        if (final_ok != 1) {
            st = TEEGNN_ERR_AUTH_FAIL;
        } else if (final_len != 0) {
            st = TEEGNN_ERR_CRYPTO;
        }
    }

    EVP_CIPHER_CTX_free(ctx);
    return st;
}
