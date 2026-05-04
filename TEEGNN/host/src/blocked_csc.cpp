#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "blocked_csc.hpp"

namespace teegnn {

#define TEEGNN_COL_PTR_BLOCK_ID UINT64_MAX
#define TEEGNN_BLOCK_AAD_LEN (8U * sizeof(uint32_t) + sizeof(uint64_t))

static int add_overflows_size_t(size_t a, size_t b) {
    return a > SIZE_MAX - b;
}

static int mul_overflows_size_t(size_t a, size_t b) {
    return b != 0 && a > SIZE_MAX / b;
}

static teegnn_status_t col_ptr_payload_size(uint32_t n_nodes, size_t *out) {
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

static teegnn_status_t row_block_payload_size(uint32_t block_size, size_t *out) {
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
    if (add_overflows_size_t(rows, values) ||
        add_overflows_size_t(rows + values, bs)) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = rows + values + bs;
    return TEEGNN_OK;
}

static uint32_t expected_num_blocks(uint32_t nnz, uint32_t block_size) {
    if (nnz == 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)nnz + block_size - 1U) / block_size);
}

static teegnn_status_t expected_blob_count(const BlockedCSCHeader *h, uint32_t *out) {
    if (h == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (h->num_blocks == UINT32_MAX) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = h->num_blocks + 1U;
    return TEEGNN_OK;
}

static teegnn_status_t validate_header(const BlockedCSCHeader *h) {
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

static teegnn_status_t compute_layout_sizes(
    const BlockedCSCHeader *h,
    uint32_t *blob_count,
    size_t *meta_offset,
    size_t *data_offset,
    size_t *total_size,
    size_t *col_payload_len,
    size_t *block_payload_len
) {
    teegnn_status_t st = validate_header(h);
    if (st != TEEGNN_OK) {
        return st;
    }

    uint32_t blobs = 0;
    st = expected_blob_count(h, &blobs);
    if (st != TEEGNN_OK) {
        return st;
    }

    size_t col_len = 0;
    size_t block_len = 0;
    st = col_ptr_payload_size(h->n_nodes, &col_len);
    if (st != TEEGNN_OK) {
        return st;
    }
    st = row_block_payload_size(h->block_size, &block_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (mul_overflows_size_t((size_t)blobs, sizeof(EncryptedBlobMeta)) ||
        mul_overflows_size_t((size_t)h->num_blocks, block_len)) {
        return TEEGNN_ERR_ALLOC;
    }

    const size_t meta_bytes = (size_t)blobs * sizeof(EncryptedBlobMeta);
    const size_t row_cipher_bytes = (size_t)h->num_blocks * block_len;
    const size_t object_header_size = offsetof(EncryptedBlockedCSC, data);
    if (add_overflows_size_t(object_header_size, meta_bytes) ||
        add_overflows_size_t(object_header_size + meta_bytes, col_len) ||
        add_overflows_size_t(object_header_size + meta_bytes + col_len, row_cipher_bytes)) {
        return TEEGNN_ERR_ALLOC;
    }

    if (blob_count != NULL) {
        *blob_count = blobs;
    }
    if (meta_offset != NULL) {
        *meta_offset = object_header_size;
    }
    if (data_offset != NULL) {
        *data_offset = object_header_size + meta_bytes;
    }
    if (total_size != NULL) {
        *total_size = object_header_size + meta_bytes + col_len + row_cipher_bytes;
    }
    if (col_payload_len != NULL) {
        *col_payload_len = col_len;
    }
    if (block_payload_len != NULL) {
        *block_payload_len = block_len;
    }
    return TEEGNN_OK;
}

static EncryptedBlobMeta *blob_meta_mut(EncryptedBlockedCSC *enc) {
    return (EncryptedBlobMeta *)((uint8_t *)enc + enc->meta_offset);
}

static const EncryptedBlobMeta *blob_meta_const(const EncryptedBlockedCSC *enc) {
    return (const EncryptedBlobMeta *)((const uint8_t *)enc + enc->meta_offset);
}

static teegnn_status_t validate_contiguous_layout(const EncryptedBlockedCSC *enc) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t expected_blobs = 0;
    size_t expected_meta_offset = 0;
    size_t expected_data_offset = 0;
    size_t expected_total_size = 0;
    size_t col_len = 0;
    size_t block_len = 0;
    teegnn_status_t st = compute_layout_sizes(
        &enc->header,
        &expected_blobs,
        &expected_meta_offset,
        &expected_data_offset,
        &expected_total_size,
        &col_len,
        &block_len
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    if (enc->blob_count != expected_blobs ||
        enc->meta_offset != (uint64_t)expected_meta_offset ||
        enc->data_offset != (uint64_t)expected_data_offset ||
        enc->total_size != (uint64_t)expected_total_size) {
        return TEEGNN_ERR_FORMAT;
    }

    const EncryptedBlobMeta *meta = blob_meta_const(enc);
    uint64_t cursor = enc->data_offset;
    for (uint32_t i = 0; i < enc->blob_count; ++i) {
        const uint64_t expected_len = (i == TEEGNN_BLOB_INDEX_COL_PTR)
            ? (uint64_t)col_len
            : (uint64_t)block_len;
        if (meta[i].ciphertext_offset != cursor ||
            meta[i].ciphertext_len != expected_len ||
            meta[i].ciphertext_offset > enc->total_size ||
            meta[i].ciphertext_len > enc->total_size - meta[i].ciphertext_offset) {
            return TEEGNN_ERR_FORMAT;
        }
        cursor += expected_len;
    }
    if (cursor != enc->total_size) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

static teegnn_status_t validate_blob_layout_at(
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index
) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t expected_blobs = 0;
    size_t expected_meta_offset = 0;
    size_t expected_data_offset = 0;
    size_t expected_total_size = 0;
    size_t col_len = 0;
    size_t block_len = 0;
    teegnn_status_t st = compute_layout_sizes(
        &enc->header,
        &expected_blobs,
        &expected_meta_offset,
        &expected_data_offset,
        &expected_total_size,
        &col_len,
        &block_len
    );
    if (st != TEEGNN_OK) {
        return st;
    }
    if (blob_index >= expected_blobs) {
        return TEEGNN_ERR_BOUNDS;
    }
    if (enc->blob_count != expected_blobs ||
        enc->meta_offset != (uint64_t)expected_meta_offset ||
        enc->data_offset != (uint64_t)expected_data_offset ||
        enc->total_size != (uint64_t)expected_total_size) {
        return TEEGNN_ERR_FORMAT;
    }

    const EncryptedBlobMeta *meta = blob_meta_const(enc);
    const EncryptedBlobMeta *m = &meta[blob_index];
    const uint64_t expected_len = (blob_index == TEEGNN_BLOB_INDEX_COL_PTR)
        ? (uint64_t)col_len
        : (uint64_t)block_len;
    const uint64_t expected_offset = (blob_index == TEEGNN_BLOB_INDEX_COL_PTR)
        ? enc->data_offset
        : enc->data_offset + (uint64_t)col_len +
              (uint64_t)(blob_index - TEEGNN_BLOB_INDEX_ROW_BLOCK(0)) * (uint64_t)block_len;

    if (m->ciphertext_offset != expected_offset ||
        m->ciphertext_len != expected_len ||
        m->ciphertext_offset > enc->total_size ||
        m->ciphertext_len > enc->total_size - m->ciphertext_offset) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

static uint32_t load_u32_slot(const uint8_t *rows, uint32_t index) {
    uint32_t value = 0;
    memcpy(&value, rows + (size_t)index * sizeof(uint32_t), sizeof(value));
    return value;
}

static double load_double_slot(const uint8_t *values, uint32_t index) {
    double value = 0.0;
    memcpy(&value, values + (size_t)index * sizeof(double), sizeof(value));
    return value;
}

static void store_u32_slot(uint8_t *rows, uint32_t index, uint32_t value) {
    memcpy(rows + (size_t)index * sizeof(uint32_t), &value, sizeof(value));
}

static void store_double_slot(uint8_t *values, uint32_t index, double value) {
    memcpy(values + (size_t)index * sizeof(double), &value, sizeof(value));
}

teegnn_status_t encrypted_blocked_csc_get_blob_view(
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    EncryptedBlobView *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    const EncryptedBlobMeta *meta = blob_meta_const(enc);
    const EncryptedBlobMeta *m = &meta[blob_index];
    out->nonce = m->nonce;
    out->tag = m->tag;
    out->ciphertext_len = m->ciphertext_len;
    out->ciphertext = (const uint8_t *)enc + m->ciphertext_offset;
    return TEEGNN_OK;
}

teegnn_status_t encrypted_blocked_csc_get_blob_mutable_view(
    EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    EncryptedBlobMutableView *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    EncryptedBlobMeta *meta = blob_meta_mut(enc);
    EncryptedBlobMeta *m = &meta[blob_index];
    out->nonce = m->nonce;
    out->tag = m->tag;
    out->ciphertext_len = m->ciphertext_len;
    out->ciphertext = (uint8_t *)enc + m->ciphertext_offset;
    return TEEGNN_OK;
}

static teegnn_status_t encrypt_blob_at_index(
    EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *payload,
    size_t payload_len,
    const uint8_t *key,
    size_t key_len
) {
    if (enc == NULL || blob_index >= enc->blob_count) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    EncryptedBlobMeta *meta = blob_meta_mut(enc);
    EncryptedBlobMeta *m = &meta[blob_index];
    if (m->ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    uint8_t aad[TEEGNN_BLOCK_AAD_LEN];
    size_t aad_len = 0;
    teegnn_status_t st = derive_nonce_simple(&enc->header, aad_block_id, m->nonce);
    if (st == TEEGNN_OK) {
        st = build_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
    }
    if (st != TEEGNN_OK) {
        return st;
    }

    return teegnn_aes_gcm_encrypt(
        key,
        key_len,
        m->nonce,
        aad,
        aad_len,
        payload,
        payload_len,
        (uint8_t *)enc + m->ciphertext_offset,
        m->tag
    );
}

static teegnn_status_t decrypt_blob_at_index(
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *payload,
    size_t payload_len
) {
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
    st = build_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
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

teegnn_status_t build_block_aad(
    const BlockedCSCHeader *h,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
) {
    if (h == NULL || aad_len == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *aad_len = TEEGNN_BLOCK_AAD_LEN;
    if (aad == NULL || aad_capacity < TEEGNN_BLOCK_AAD_LEN) {
        return TEEGNN_ERR_BOUNDS;
    }

    size_t off = 0;
#define COPY_FIELD(field) do { \
        memcpy(aad + off, &(h->field), sizeof(h->field)); \
        off += sizeof(h->field); \
    } while (0)
    COPY_FIELD(graph_id);
    COPY_FIELD(graph_version);
    COPY_FIELD(layer_id);
    COPY_FIELD(layout_version);
    COPY_FIELD(n_nodes);
    COPY_FIELD(nnz);
    COPY_FIELD(block_size);
    COPY_FIELD(num_blocks);
#undef COPY_FIELD
    memcpy(aad + off, &block_id, sizeof(block_id));
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

teegnn_status_t derive_nonce_simple(
    const BlockedCSCHeader *h,
    uint64_t block_id,
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN]
) {
    if (h == NULL || nonce == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t context_hash = 2166136261U;
    context_hash = fnv1a_mix_u32(context_hash, h->graph_id);
    context_hash = fnv1a_mix_u32(context_hash, h->graph_version);
    context_hash = fnv1a_mix_u32(context_hash, h->layer_id);
    context_hash = fnv1a_mix_u32(context_hash, h->layout_version);
    context_hash = fnv1a_mix_u32(context_hash, h->n_nodes);
    context_hash = fnv1a_mix_u32(context_hash, h->nnz);
    context_hash = fnv1a_mix_u32(context_hash, h->block_size);
    context_hash = fnv1a_mix_u32(context_hash, h->num_blocks);

    memcpy(nonce, &context_hash, sizeof(context_hash));
    memcpy(nonce + sizeof(context_hash), &block_id, sizeof(block_id));
    return TEEGNN_OK;
}

teegnn_status_t blocked_csc_encrypt(
    const CSCGraph *A,
    uint32_t graph_id,
    uint32_t graph_version,
    uint32_t layer_id,
    uint32_t layout_version,
    uint32_t block_size,
    const uint8_t *key,
    size_t key_len,
    EncryptedBlockedCSC **out
) {
    if (out == NULL || block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *out = NULL;

    teegnn_status_t st;

    BlockedCSCHeader header;
    memset(&header, 0, sizeof(header));
    header.graph_id = graph_id;
    header.graph_version = graph_version;
    header.layer_id = layer_id;
    header.layout_version = layout_version;
    header.n_nodes = A->n_nodes;
    header.nnz = A->nnz;
    header.block_size = block_size;
    header.num_blocks = expected_num_blocks(A->nnz, block_size);

    uint32_t blob_count = 0;
    size_t meta_offset = 0;
    size_t data_offset = 0;
    size_t total_size = 0;
    size_t col_payload_len = 0;
    size_t block_payload_len = 0;
    st = compute_layout_sizes(
        &header,
        &blob_count,
        &meta_offset,
        &data_offset,
        &total_size,
        &col_payload_len,
        &block_payload_len
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    EncryptedBlockedCSC *enc = (EncryptedBlockedCSC *)calloc(1U, total_size);
    if (enc == NULL) {
        return TEEGNN_ERR_ALLOC;
    }
    enc->header = header;
    enc->blob_count = blob_count;
    enc->reserved = 0;
    enc->meta_offset = (uint64_t)meta_offset;
    enc->data_offset = (uint64_t)data_offset;
    enc->total_size = (uint64_t)total_size;

    EncryptedBlobMeta *meta = blob_meta_mut(enc);
    uint64_t cursor = enc->data_offset;
    meta[TEEGNN_BLOB_INDEX_COL_PTR].ciphertext_offset = cursor;
    meta[TEEGNN_BLOB_INDEX_COL_PTR].ciphertext_len = (uint64_t)col_payload_len;
    cursor += (uint64_t)col_payload_len;

    for (uint32_t block_id = 0; block_id < header.num_blocks; ++block_id) {
        const uint32_t blob_index = TEEGNN_BLOB_INDEX_ROW_BLOCK(block_id);
        meta[blob_index].ciphertext_offset = cursor;
        meta[blob_index].ciphertext_len = (uint64_t)block_payload_len;
        cursor += (uint64_t)block_payload_len;
    }

    st = encrypt_blob_at_index(
        enc,
        TEEGNN_BLOB_INDEX_COL_PTR,
        TEEGNN_COL_PTR_BLOCK_ID,
        (const uint8_t *)A->col_ptr,
        col_payload_len,
        key,
        key_len
    );
    if (st != TEEGNN_OK) {
        encrypted_blocked_csc_free(enc);
        return st;
    }

    for (uint32_t block_id = 0; block_id < header.num_blocks; ++block_id) {
        uint8_t *payload = (uint8_t *)calloc(1U, block_payload_len);
        if (payload == NULL) {
            encrypted_blocked_csc_free(enc);
            return TEEGNN_ERR_ALLOC;
        }

        uint8_t *rows = payload;
        uint8_t *values = rows + (size_t)block_size * sizeof(uint32_t);
        uint8_t *valid = values + (size_t)block_size * sizeof(double);

        for (uint32_t t = 0; t < block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * block_size + t;
            if (global_pos < A->nnz) {
                store_u32_slot(rows, t, A->row_idx[global_pos]);
                store_double_slot(values, t, A->values[global_pos]);
                valid[t] = 1U;
            }
        }

        st = encrypt_blob_at_index(
            enc,
            TEEGNN_BLOB_INDEX_ROW_BLOCK(block_id),
            block_id,
            payload,
            block_payload_len,
            key,
            key_len
        );
        free(payload);
        if (st != TEEGNN_OK) {
            encrypted_blocked_csc_free(enc);
            return st;
        }
    }

    *out = enc;
    return TEEGNN_OK;
}

teegnn_status_t blocked_csc_decrypt_full(
    const EncryptedBlockedCSC *enc,
    const uint8_t *key,
    size_t key_len,
    CSCGraph *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    teegnn_status_t st = validate_contiguous_layout(enc);
    if (st != TEEGNN_OK) {
        return st;
    }

    CSCGraph tmp_graph;
    memset(&tmp_graph, 0, sizeof(tmp_graph));
    st = csc_graph_alloc(&tmp_graph, enc->header.n_nodes, enc->header.nnz);
    if (st != TEEGNN_OK) {
        return st;
    }

    size_t col_payload_len = 0;
    st = col_ptr_payload_size(enc->header.n_nodes, &col_payload_len);
    if (st == TEEGNN_OK) {
        st = decrypt_blob_at_index(
            enc,
            TEEGNN_BLOB_INDEX_COL_PTR,
            TEEGNN_COL_PTR_BLOCK_ID,
            key,
            key_len,
            (uint8_t *)tmp_graph.col_ptr,
            col_payload_len
        );
    }
    if (st != TEEGNN_OK) {
        csc_graph_free(&tmp_graph);
        return st;
    }

    if (tmp_graph.col_ptr[0] != 0 ||
        tmp_graph.col_ptr[enc->header.n_nodes] != enc->header.nnz) {
        csc_graph_free(&tmp_graph);
        return TEEGNN_ERR_FORMAT;
    }
    for (uint32_t col = 0; col < enc->header.n_nodes; ++col) {
        if (tmp_graph.col_ptr[col] > tmp_graph.col_ptr[col + 1] ||
            tmp_graph.col_ptr[col + 1] > enc->header.nnz) {
            csc_graph_free(&tmp_graph);
            return TEEGNN_ERR_FORMAT;
        }
    }

    size_t block_payload_len = 0;
    st = row_block_payload_size(enc->header.block_size, &block_payload_len);
    if (st != TEEGNN_OK) {
        csc_graph_free(&tmp_graph);
        return st;
    }

    for (uint32_t block_id = 0; block_id < enc->header.num_blocks; ++block_id) {
        uint8_t *payload = (uint8_t *)malloc(block_payload_len);
        if (payload == NULL) {
            csc_graph_free(&tmp_graph);
            return TEEGNN_ERR_ALLOC;
        }

        st = decrypt_blob_at_index(
            enc,
            TEEGNN_BLOB_INDEX_ROW_BLOCK(block_id),
            block_id,
            key,
            key_len,
            payload,
            block_payload_len
        );
        if (st != TEEGNN_OK) {
            free(payload);
            csc_graph_free(&tmp_graph);
            return st;
        }

        const uint8_t *rows = payload;
        const uint8_t *values = rows + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *valid = values + (size_t)enc->header.block_size * sizeof(double);

        for (uint32_t t = 0; t < enc->header.block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * enc->header.block_size + t;
            const uint8_t is_valid = valid[t];
            if (is_valid > 1U) {
                free(payload);
                csc_graph_free(&tmp_graph);
                return TEEGNN_ERR_FORMAT;
            }
            if (global_pos < enc->header.nnz) {
                if (is_valid != 1U) {
                    free(payload);
                    csc_graph_free(&tmp_graph);
                    return TEEGNN_ERR_FORMAT;
                }
                const uint32_t row = load_u32_slot(rows, t);
                const double value = load_double_slot(values, t);
                tmp_graph.row_idx[global_pos] = row;
                tmp_graph.values[global_pos] = value;
            } else {
                const uint32_t row = load_u32_slot(rows, t);
                const double value = load_double_slot(values, t);
                if (is_valid != 0U || row != 0U || value != 0.0) {
                    free(payload);
                    csc_graph_free(&tmp_graph);
                    return TEEGNN_ERR_FORMAT;
                }
            }
        }
        free(payload);
    }

    *out = tmp_graph;
    return TEEGNN_OK;
}

void encrypted_blocked_csc_free(
    EncryptedBlockedCSC *enc
) {
    free(enc);
}

}  // namespace teegnn