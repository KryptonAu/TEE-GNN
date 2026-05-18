#include "blocked_edge_list.h"

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define TEEGNN_EDGE_BLOCK_AAD_LEN (8U * sizeof(uint32_t) + sizeof(uint64_t))

static int add_overflows_size_t(size_t a, size_t b) {
    return a > SIZE_MAX - b;
}

static int mul_overflows_size_t(size_t a, size_t b) {
    return b != 0 && a > SIZE_MAX / b;
}

static teegnn_status_t edge_block_payload_size(uint32_t block_size, size_t *out) {
    if (out == NULL || block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    const size_t bs = (size_t)block_size;
    if (mul_overflows_size_t(bs, sizeof(uint32_t)) ||
        mul_overflows_size_t(bs, sizeof(float))) {
        return TEEGNN_ERR_ALLOC;
    }

    const size_t srcs = bs * sizeof(uint32_t);
    const size_t dsts = bs * sizeof(uint32_t);
    const size_t values = bs * sizeof(float);
    if (add_overflows_size_t(dsts, srcs) ||
        add_overflows_size_t(dsts + srcs, values) ||
        add_overflows_size_t(dsts + srcs + values, bs)) {
        return TEEGNN_ERR_ALLOC;
    }

    *out = dsts + srcs + values + bs;
    return TEEGNN_OK;
}

static uint32_t expected_num_blocks(uint32_t m_edges, uint32_t block_size) {
    if (m_edges == 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)m_edges + block_size - 1U) / block_size);
}

static teegnn_status_t validate_header(const BlockedEdgeListHeader *h) {
    if (h == NULL || h->block_size == 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->n_nodes == 0 && h->m_edges != 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (h->num_blocks != expected_num_blocks(h->m_edges, h->block_size)) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

static teegnn_status_t compute_layout_sizes(
    const BlockedEdgeListHeader *h,
    uint32_t *blob_count,
    size_t *meta_offset,
    size_t *data_offset,
    size_t *total_size,
    size_t *block_payload_len
) {
    teegnn_status_t st = validate_header(h);
    if (st != TEEGNN_OK) {
        return st;
    }

    size_t block_len = 0;
    st = edge_block_payload_size(h->block_size, &block_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (mul_overflows_size_t((size_t)h->num_blocks, sizeof(EncryptedEdgeBlobMeta)) ||
        mul_overflows_size_t((size_t)h->num_blocks, block_len)) {
        return TEEGNN_ERR_ALLOC;
    }

    const size_t meta_bytes = (size_t)h->num_blocks * sizeof(EncryptedEdgeBlobMeta);
    const size_t block_cipher_bytes = (size_t)h->num_blocks * block_len;
    const size_t object_header_size = offsetof(EncryptedBlockedEdgeList, data);
    if (add_overflows_size_t(object_header_size, meta_bytes) ||
        add_overflows_size_t(object_header_size + meta_bytes, block_cipher_bytes)) {
        return TEEGNN_ERR_ALLOC;
    }

    if (blob_count != NULL) {
        *blob_count = h->num_blocks;
    }
    if (meta_offset != NULL) {
        *meta_offset = object_header_size;
    }
    if (data_offset != NULL) {
        *data_offset = object_header_size + meta_bytes;
    }
    if (total_size != NULL) {
        *total_size = object_header_size + meta_bytes + block_cipher_bytes;
    }
    if (block_payload_len != NULL) {
        *block_payload_len = block_len;
    }
    return TEEGNN_OK;
}

static EncryptedEdgeBlobMeta *blob_meta_mut(EncryptedBlockedEdgeList *enc) {
    return (EncryptedEdgeBlobMeta *)((uint8_t *)enc + enc->meta_offset);
}

static const EncryptedEdgeBlobMeta *blob_meta_const(const EncryptedBlockedEdgeList *enc) {
    return (const EncryptedEdgeBlobMeta *)((const uint8_t *)enc + enc->meta_offset);
}

static teegnn_status_t validate_contiguous_layout(const EncryptedBlockedEdgeList *enc) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t expected_blobs = 0;
    size_t expected_meta_offset = 0;
    size_t expected_data_offset = 0;
    size_t expected_total_size = 0;
    size_t block_len = 0;
    teegnn_status_t st = compute_layout_sizes(
        &enc->header,
        &expected_blobs,
        &expected_meta_offset,
        &expected_data_offset,
        &expected_total_size,
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

    const EncryptedEdgeBlobMeta *meta = blob_meta_const(enc);
    uint64_t cursor = enc->data_offset;
    for (uint32_t i = 0; i < enc->blob_count; ++i) {
        if (meta[i].ciphertext_offset != cursor ||
            meta[i].ciphertext_len != (uint64_t)block_len ||
            meta[i].ciphertext_offset > enc->total_size ||
            meta[i].ciphertext_len > enc->total_size - meta[i].ciphertext_offset) {
            return TEEGNN_ERR_FORMAT;
        }
        cursor += (uint64_t)block_len;
    }
    if (cursor != enc->total_size) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

static teegnn_status_t validate_blob_layout_at(
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index
) {
    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    uint32_t expected_blobs = 0;
    size_t expected_meta_offset = 0;
    size_t expected_data_offset = 0;
    size_t expected_total_size = 0;
    size_t block_len = 0;
    teegnn_status_t st = compute_layout_sizes(
        &enc->header,
        &expected_blobs,
        &expected_meta_offset,
        &expected_data_offset,
        &expected_total_size,
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

    const EncryptedEdgeBlobMeta *meta = blob_meta_const(enc);
    const EncryptedEdgeBlobMeta *m = &meta[blob_index];
    const uint64_t expected_offset =
        enc->data_offset + (uint64_t)blob_index * (uint64_t)block_len;
    if (m->ciphertext_offset != expected_offset ||
        m->ciphertext_len != (uint64_t)block_len ||
        m->ciphertext_offset > enc->total_size ||
        m->ciphertext_len > enc->total_size - m->ciphertext_offset) {
        return TEEGNN_ERR_FORMAT;
    }
    return TEEGNN_OK;
}

static uint32_t load_u32_slot(const uint8_t *data, uint32_t index) {
    uint32_t value = 0;
    memcpy(&value, data + (size_t)index * sizeof(uint32_t), sizeof(value));
    return value;
}

static double load_float_slot_as_double(const uint8_t *data, uint32_t index) {
    float value = 0.0f;
    memcpy(&value, data + (size_t)index * sizeof(float), sizeof(value));
    return (double)value;
}

static void store_u32_slot(uint8_t *data, uint32_t index, uint32_t value) {
    memcpy(data + (size_t)index * sizeof(uint32_t), &value, sizeof(value));
}

static void store_float_slot_from_double(uint8_t *data, uint32_t index, double value) {
    float wire_value = (float)value;
    memcpy(data + (size_t)index * sizeof(float), &wire_value, sizeof(wire_value));
}

teegnn_status_t encrypted_blocked_edge_list_get_blob_view(
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    EncryptedEdgeBlobView *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    const EncryptedEdgeBlobMeta *meta = blob_meta_const(enc);
    const EncryptedEdgeBlobMeta *m = &meta[blob_index];
    out->nonce = m->nonce;
    out->tag = m->tag;
    out->ciphertext_len = m->ciphertext_len;
    out->ciphertext = (const uint8_t *)enc + m->ciphertext_offset;
    return TEEGNN_OK;
}

teegnn_status_t encrypted_blocked_edge_list_get_blob_mutable_view(
    EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    EncryptedEdgeBlobMutableView *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    EncryptedEdgeBlobMeta *meta = blob_meta_mut(enc);
    EncryptedEdgeBlobMeta *m = &meta[blob_index];
    out->nonce = m->nonce;
    out->tag = m->tag;
    out->ciphertext_len = m->ciphertext_len;
    out->ciphertext = (uint8_t *)enc + m->ciphertext_offset;
    return TEEGNN_OK;
}

teegnn_status_t build_edge_block_aad(
    const BlockedEdgeListHeader *h,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
) {
    if (h == NULL || aad_len == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *aad_len = TEEGNN_EDGE_BLOCK_AAD_LEN;
    if (aad == NULL || aad_capacity < TEEGNN_EDGE_BLOCK_AAD_LEN) {
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
    COPY_FIELD(m_edges);
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

teegnn_status_t derive_edge_nonce_simple(
    const BlockedEdgeListHeader *h,
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
    context_hash = fnv1a_mix_u32(context_hash, h->m_edges);
    context_hash = fnv1a_mix_u32(context_hash, h->block_size);
    context_hash = fnv1a_mix_u32(context_hash, h->num_blocks);

    memcpy(nonce, &context_hash, sizeof(context_hash));
    memcpy(nonce + sizeof(context_hash), &block_id, sizeof(block_id));
    return TEEGNN_OK;
}

static teegnn_status_t encrypt_blob_at_index(
    EncryptedBlockedEdgeList *enc,
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

    EncryptedEdgeBlobMeta *meta = blob_meta_mut(enc);
    EncryptedEdgeBlobMeta *m = &meta[blob_index];
    if (m->ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    uint8_t aad[TEEGNN_EDGE_BLOCK_AAD_LEN];
    size_t aad_len = 0;
    teegnn_status_t st = derive_edge_nonce_simple(&enc->header, aad_block_id, m->nonce);
    if (st == TEEGNN_OK) {
        st = build_edge_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
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
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *payload,
    size_t payload_len
) {
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
    st = build_edge_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
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

teegnn_status_t blocked_edge_list_encrypt(
    const EdgeList *A,
    uint32_t graph_id,
    uint32_t graph_version,
    uint32_t layer_id,
    uint32_t layout_version,
    uint32_t block_size,
    const uint8_t *key,
    size_t key_len,
    EncryptedBlockedEdgeList **out
) {
    if (out == NULL || block_size == 0) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *out = NULL;

    teegnn_status_t st = edge_list_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }

    BlockedEdgeListHeader header;
    memset(&header, 0, sizeof(header));
    header.graph_id = graph_id;
    header.graph_version = graph_version;
    header.layer_id = layer_id;
    header.layout_version = layout_version;
    header.n_nodes = A->n_nodes;
    header.m_edges = A->m_edges;
    header.block_size = block_size;
    header.num_blocks = expected_num_blocks(A->m_edges, block_size);

    uint32_t blob_count = 0;
    size_t meta_offset = 0;
    size_t data_offset = 0;
    size_t total_size = 0;
    size_t block_payload_len = 0;
    st = compute_layout_sizes(
        &header,
        &blob_count,
        &meta_offset,
        &data_offset,
        &total_size,
        &block_payload_len
    );
    if (st != TEEGNN_OK) {
        return st;
    }

    EncryptedBlockedEdgeList *enc = (EncryptedBlockedEdgeList *)calloc(1U, total_size);
    if (enc == NULL) {
        return TEEGNN_ERR_ALLOC;
    }
    enc->header = header;
    enc->blob_count = blob_count;
    enc->reserved = 0;
    enc->meta_offset = (uint64_t)meta_offset;
    enc->data_offset = (uint64_t)data_offset;
    enc->total_size = (uint64_t)total_size;

    EncryptedEdgeBlobMeta *meta = blob_meta_mut(enc);
    uint64_t cursor = enc->data_offset;
    for (uint32_t block_id = 0; block_id < header.num_blocks; ++block_id) {
        const uint32_t blob_index = TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id);
        meta[blob_index].ciphertext_offset = cursor;
        meta[blob_index].ciphertext_len = (uint64_t)block_payload_len;
        cursor += (uint64_t)block_payload_len;
    }

    for (uint32_t block_id = 0; block_id < header.num_blocks; ++block_id) {
        uint8_t *payload = (uint8_t *)calloc(1U, block_payload_len);
        if (payload == NULL) {
            encrypted_blocked_edge_list_free(enc);
            return TEEGNN_ERR_ALLOC;
        }

        uint8_t *dsts = payload;
        uint8_t *srcs = dsts + (size_t)block_size * sizeof(uint32_t);
        uint8_t *values = srcs + (size_t)block_size * sizeof(uint32_t);
        uint8_t *valid = values + (size_t)block_size * sizeof(float);

        for (uint32_t t = 0; t < block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * block_size + t;
            if (global_pos < A->m_edges) {
                const Edge *e = &A->e_ptr[global_pos];
                store_u32_slot(dsts, t, (uint32_t)e->dst);
                store_u32_slot(srcs, t, (uint32_t)e->src);
                store_float_slot_from_double(values, t, e->value);
                valid[t] = 1U;
            }
        }

        st = encrypt_blob_at_index(
            enc,
            TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id),
            block_id,
            payload,
            block_payload_len,
            key,
            key_len
        );
        free(payload);
        if (st != TEEGNN_OK) {
            encrypted_blocked_edge_list_free(enc);
            return st;
        }
    }

    *out = enc;
    return TEEGNN_OK;
}

teegnn_status_t blocked_edge_list_decrypt_full(
    const EncryptedBlockedEdgeList *enc,
    const uint8_t *key,
    size_t key_len,
    EdgeList *out
) {
    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    teegnn_status_t st = validate_contiguous_layout(enc);
    if (st != TEEGNN_OK) {
        return st;
    }

    EdgeList tmp;
    memset(&tmp, 0, sizeof(tmp));
    st = edge_list_alloc(&tmp, enc->header.n_nodes, enc->header.m_edges);
    if (st != TEEGNN_OK) {
        return st;
    }

    size_t block_payload_len = 0;
    st = edge_block_payload_size(enc->header.block_size, &block_payload_len);
    if (st != TEEGNN_OK) {
        edge_list_free(&tmp);
        return st;
    }

    for (uint32_t block_id = 0; block_id < enc->header.num_blocks; ++block_id) {
        uint8_t *payload = (uint8_t *)malloc(block_payload_len);
        if (payload == NULL) {
            edge_list_free(&tmp);
            return TEEGNN_ERR_ALLOC;
        }

        st = decrypt_blob_at_index(
            enc,
            TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id),
            block_id,
            key,
            key_len,
            payload,
            block_payload_len
        );
        if (st != TEEGNN_OK) {
            free(payload);
            edge_list_free(&tmp);
            return st;
        }

        const uint8_t *dsts = payload;
        const uint8_t *srcs = dsts + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *values = srcs + (size_t)enc->header.block_size * sizeof(uint32_t);
        const uint8_t *valid = values + (size_t)enc->header.block_size * sizeof(float);

        for (uint32_t t = 0; t < enc->header.block_size; ++t) {
            const uint64_t global_pos = (uint64_t)block_id * enc->header.block_size + t;
            const uint8_t is_valid = valid[t];
            if (is_valid > 1U) {
                free(payload);
                edge_list_free(&tmp);
                return TEEGNN_ERR_FORMAT;
            }
            if (global_pos < enc->header.m_edges) {
                if (is_valid != 1U) {
                    free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_FORMAT;
                }
                const uint32_t dst = load_u32_slot(dsts, t);
                const uint32_t src = load_u32_slot(srcs, t);
                if (src >= enc->header.n_nodes || dst >= enc->header.n_nodes ||
                    src > (uint32_t)INT32_MAX || dst > (uint32_t)INT32_MAX) {
                    free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_BOUNDS;
                }
                tmp.e_ptr[global_pos].src = (int32_t)src;
                tmp.e_ptr[global_pos].dst = (int32_t)dst;
                tmp.e_ptr[global_pos].value = load_float_slot_as_double(values, t);
            } else {
                const uint32_t dst = load_u32_slot(dsts, t);
                const uint32_t src = load_u32_slot(srcs, t);
                const double value = load_float_slot_as_double(values, t);
                if (is_valid != 0U || dst != 0U || src != 0U || value != 0.0) {
                    free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_FORMAT;
                }
            }
        }
        free(payload);
    }

    st = edge_list_validate(&tmp);
    if (st != TEEGNN_OK) {
        edge_list_free(&tmp);
        return st;
    }

    *out = tmp;
    return TEEGNN_OK;
}

void encrypted_blocked_edge_list_free(
    EncryptedBlockedEdgeList *enc
) {
    free(enc);
}
