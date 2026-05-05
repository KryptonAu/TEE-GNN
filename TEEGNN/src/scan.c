#include "scan.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int mul_overflows_size_t(size_t a, size_t b) {
    return b != 0 && a > SIZE_MAX / b;
}

uint32_t expected_num_blocks(uint32_t nnz, uint32_t block_size) {
    if (nnz == 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)nnz + block_size - 1U) / block_size);
}

teegnn_status_t validate_header_for_scan(const BlockedCSCHeader *h) {
    if (h == NULL || h->block_size == 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->n_nodes == 0 && h->nnz != 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->num_blocks != expected_num_blocks(h->nnz, h->block_size)) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

teegnn_status_t col_ptr_payload_size(uint32_t n_nodes, size_t *out) {
    if (out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    const size_t count = (size_t)n_nodes + 1U;
    if (mul_overflows_size_t(count, sizeof(uint32_t))) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = count * sizeof(uint32_t);
    return TEEGNN_OK;
}

teegnn_status_t row_block_payload_size(uint32_t block_size, size_t *out) {
    if (out == NULL || block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    const size_t bs = (size_t)block_size;
    if (mul_overflows_size_t(bs, sizeof(uint32_t)) ||
        mul_overflows_size_t(bs, sizeof(double))) {
        return TEEGNN_ERR_ALLOC;
    }
    const size_t rows = bs * sizeof(uint32_t);
    const size_t values = bs * sizeof(double);
    if (rows > SIZE_MAX - values || rows + values > SIZE_MAX - bs) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = rows + values + bs;
    return TEEGNN_OK;
}

teegnn_status_t decrypt_blob_for_scan(
    const BlockedCSCHeader *h,
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *payload,
    size_t payload_len
) {
    if (h == NULL || enc == NULL || payload == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    EncryptedBlobView view;
    teegnn_status_t st = encrypted_blocked_csc_get_blob_view(enc, blob_index, &view);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (view.ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    uint8_t aad[TEEGNN_BLOCK_AAD_LEN];
    size_t aad_len = 0;
    st = build_block_aad(h, aad_block_id, aad, sizeof(aad), &aad_len);
    if (st != TEEGNN_OK) {
        return st;
    }
    return teegnn_aes_gcm_decrypt(
        key,
        key_len,
        view.nonce,
        aad,
        aad_len,
        view.ciphertext,
        payload_len,
        view.tag,
        payload
    );
}

uint32_t load_u32_slot(const uint8_t *rows, uint32_t index) {
    uint32_t value = 0;
    memcpy(&value, rows + (size_t)index * sizeof(uint32_t), sizeof(value));
    return value;
}

double load_double_slot(const uint8_t *values, uint32_t index) {
    double value = 0.0;
    memcpy(&value, values + (size_t)index * sizeof(double), sizeof(value));
    return value;
}

teegnn_status_t validate_col_ptr(
    const uint32_t *col_ptr,
    uint32_t n_nodes,
    uint32_t nnz
) {
    if (col_ptr == NULL || col_ptr[0] != 0 || col_ptr[n_nodes] != nnz) {
        return TEEGNN_ERR_FORMAT;
    }
    for (uint32_t col = 0; col < n_nodes; ++col) {
        if (col_ptr[col] > col_ptr[col + 1] || col_ptr[col + 1] > nnz) {
            return TEEGNN_ERR_FORMAT;
        }
    }
    return TEEGNN_OK;
}
