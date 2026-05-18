#include "scan.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

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

uint32_t expected_edge_num_blocks(uint32_t m_edges, uint32_t block_size) {
    if (m_edges == 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)m_edges + block_size - 1U) / block_size);
}

teegnn_status_t validate_edge_header_for_scan(const BlockedEdgeListHeader *h) {
    if (h == NULL || h->block_size == 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->n_nodes == 0 && h->m_edges != 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->num_blocks != expected_edge_num_blocks(h->m_edges, h->block_size)) {
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

teegnn_status_t edge_block_payload_size(uint32_t block_size, size_t *out) {
    if (out == NULL || block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    const size_t bs = (size_t)block_size;
    if (mul_overflows_size_t(bs, sizeof(uint32_t)) ||
        mul_overflows_size_t(bs, sizeof(float))) {
        return TEEGNN_ERR_ALLOC;
    }
    const size_t dsts = bs * sizeof(uint32_t);
    const size_t srcs = bs * sizeof(uint32_t);
    const size_t values = bs * sizeof(float);
    if (dsts > SIZE_MAX - srcs ||
        dsts + srcs > SIZE_MAX - values ||
        dsts + srcs + values > SIZE_MAX - bs) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = dsts + srcs + values + bs;
    return TEEGNN_OK;
}

teegnn_status_t validate_edge_object_layout_for_scan(
    const EncryptedBlockedEdgeList *enc
) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    size_t block_len = 0;
    teegnn_status_t st = edge_block_payload_size(enc->header.block_size, &block_len);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (mul_overflows_size_t((size_t)enc->header.num_blocks, sizeof(EncryptedEdgeBlobMeta)) ||
        mul_overflows_size_t((size_t)enc->header.num_blocks, block_len)) {
        return TEEGNN_ERR_ALLOC;
    }

    const size_t meta_bytes =
        (size_t)enc->header.num_blocks * sizeof(EncryptedEdgeBlobMeta);
    const size_t block_cipher_bytes = (size_t)enc->header.num_blocks * block_len;
    const size_t object_header_size = offsetof(EncryptedBlockedEdgeList, data);
    if (object_header_size > SIZE_MAX - meta_bytes ||
        object_header_size + meta_bytes > SIZE_MAX - block_cipher_bytes) {
        return TEEGNN_ERR_ALLOC;
    }

    const size_t expected_data_offset = object_header_size + meta_bytes;
    const size_t expected_total_size = expected_data_offset + block_cipher_bytes;
    if (enc->blob_count != enc->header.num_blocks ||
        enc->meta_offset != (uint64_t)object_header_size ||
        enc->data_offset != (uint64_t)expected_data_offset ||
        enc->total_size != (uint64_t)expected_total_size) {
        return TEEGNN_ERR_FORMAT;
    }
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

teegnn_status_t decrypt_edge_blob_for_scan(
    const BlockedEdgeListHeader *h,
    const EncryptedBlockedEdgeList *enc,
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

    EncryptedEdgeBlobView view;
    teegnn_status_t st = encrypted_blocked_edge_list_get_blob_view(enc, blob_index, &view);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (view.ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    uint8_t aad[TEEGNN_EDGE_BLOCK_AAD_LEN];
    size_t aad_len = 0;
    st = build_edge_block_aad(h, aad_block_id, aad, sizeof(aad), &aad_len);
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
    TEE_MemMove(&value, rows + (size_t)index * sizeof(uint32_t), sizeof(value));
    return value;
}

double load_float_slot_as_double(const uint8_t *values, uint32_t index) {
    float value = 0.0f;
    TEE_MemMove(&value, values + (size_t)index * sizeof(float), sizeof(value));
    return (double)value;
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

static uint32_t fnv1a_mix_u32(uint32_t hash, uint32_t value) {
    const uint8_t *bytes = (const uint8_t *)&value;
    for (size_t i = 0; i < sizeof(value); ++i) {
        hash ^= bytes[i];
        hash *= 16777619U;
    }
    return hash;
}

teegnn_status_t matrix_block_layout(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t *blocks,
    size_t *payload_len,
    size_t *block_stride
) {
    if (row_block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    const uint32_t n_blocks = rows == 0
        ? 0U
        : (uint32_t)(((uint64_t)rows + row_block_size - 1U) / row_block_size);
    if (mul_overflows_size_t((size_t)row_block_size, (size_t)cols)) {
        return TEEGNN_ERR_ALLOC;
    }
    const size_t elements = (size_t)row_block_size * (size_t)cols;
    if (mul_overflows_size_t(elements, sizeof(float))) {
        return TEEGNN_ERR_ALLOC;
    }
    const size_t block_payload_len = elements * sizeof(float);

    if (blocks != NULL) {
        *blocks = n_blocks;
    }
    if (payload_len != NULL) {
        *payload_len = block_payload_len;
    }
    if (block_stride != NULL) {
        *block_stride = TEEGNN_GCM_NONCE_LEN + TEEGNN_GCM_TAG_LEN + block_payload_len;
    }
    return TEEGNN_OK;
}

teegnn_status_t build_matrix_block_aad(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
) {
    if (aad_len == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *aad_len = TEEGNN_MATRIX_BLOCK_AAD_LEN;
    if (aad == NULL || aad_capacity < TEEGNN_MATRIX_BLOCK_AAD_LEN) {
        return TEEGNN_ERR_BOUNDS;
    }

    size_t off = 0;
    TEE_MemMove(aad + off, &row_block_size, sizeof(row_block_size));
    off += sizeof(row_block_size);
    TEE_MemMove(aad + off, &rows, sizeof(rows));
    off += sizeof(rows);
    TEE_MemMove(aad + off, &cols, sizeof(cols));
    off += sizeof(cols);
    TEE_MemMove(aad + off, &blocks, sizeof(blocks));
    off += sizeof(blocks);
    TEE_MemMove(aad + off, &block_id, sizeof(block_id));
    return TEEGNN_OK;
}

teegnn_status_t derive_matrix_nonce_simple(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks,
    uint64_t block_id,
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN]
) {
    if (nonce == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t context_hash = 2166136261U;
    context_hash = fnv1a_mix_u32(context_hash, row_block_size);
    context_hash = fnv1a_mix_u32(context_hash, rows);
    context_hash = fnv1a_mix_u32(context_hash, cols);
    context_hash = fnv1a_mix_u32(context_hash, blocks);

    TEE_MemMove(nonce, &context_hash, sizeof(context_hash));
    TEE_MemMove(nonce + sizeof(context_hash), &block_id, sizeof(block_id));
    return TEEGNN_OK;
}