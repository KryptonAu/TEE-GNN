#include "blocked_edge_list.h"

#include <stddef.h>
#include <stdint.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

#define TEEGNN_EDGE_BLOCK_AAD_LEN (8U * sizeof(uint32_t) + sizeof(uint64_t))
#define TEEGNN_MATRIX_BLOCK_AAD_LEN (4U * sizeof(uint32_t) + sizeof(uint64_t))

static int teegnn_add_overflows_size_t(size_t a, size_t b)
{
    return a > SIZE_MAX - b;
}

static int teegnn_mul_overflows_size_t(size_t a, size_t b)
{
    return b != 0U && a > SIZE_MAX / b;
}

static teegnn_status_t edge_block_payload_size(uint32_t block_size, size_t *out)
{
    size_t bs;
    size_t dsts;
    size_t srcs;
    size_t values;
    size_t total;

    if (out == NULL || block_size == 0U) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    bs = (size_t)block_size;
    if (teegnn_mul_overflows_size_t(bs, sizeof(uint32_t)) ||
        teegnn_mul_overflows_size_t(bs, sizeof(double))) {
        return TEEGNN_ERR_ALLOC;
    }

    dsts = bs * sizeof(uint32_t);
    srcs = bs * sizeof(uint32_t);
    values = bs * sizeof(double);

    if (teegnn_add_overflows_size_t(dsts, srcs)) {
        return TEEGNN_ERR_ALLOC;
    }
    total = dsts + srcs;

    if (teegnn_add_overflows_size_t(total, values)) {
        return TEEGNN_ERR_ALLOC;
    }
    total += values;

    if (teegnn_add_overflows_size_t(total, bs)) {
        return TEEGNN_ERR_ALLOC;
    }
    total += bs;

    *out = total;
    return TEEGNN_OK;
}

static uint32_t expected_num_blocks(uint32_t m_edges, uint32_t block_size)
{
    if (m_edges == 0U) {
        return 0U;
    }
    return (uint32_t)(((uint64_t)m_edges + (uint64_t)block_size - 1U) /
                      (uint64_t)block_size);
}

static teegnn_status_t validate_header(const BlockedEdgeListHeader *h)
{
    if (h == NULL || h->block_size == 0U) {
        return TEEGNN_ERR_FORMAT;
    }

    if (h->n_nodes == 0U && h->m_edges != 0U) {
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
    teegnn_status_t st;
    size_t block_len = 0U;
    size_t meta_bytes;
    size_t block_cipher_bytes;
    size_t object_header_size;
    size_t data_off;
    size_t total;

    st = validate_header(h);
    if (st != TEEGNN_OK) {
        return st;
    }

    st = edge_block_payload_size(h->block_size, &block_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (teegnn_mul_overflows_size_t((size_t)h->num_blocks,
                                    sizeof(EncryptedEdgeBlobMeta)) ||
        teegnn_mul_overflows_size_t((size_t)h->num_blocks, block_len)) {
        return TEEGNN_ERR_ALLOC;
    }

    meta_bytes = (size_t)h->num_blocks * sizeof(EncryptedEdgeBlobMeta);
    block_cipher_bytes = (size_t)h->num_blocks * block_len;
    object_header_size = offsetof(EncryptedBlockedEdgeList, data);

    if (teegnn_add_overflows_size_t(object_header_size, meta_bytes)) {
        return TEEGNN_ERR_ALLOC;
    }
    data_off = object_header_size + meta_bytes;

    if (teegnn_add_overflows_size_t(data_off, block_cipher_bytes)) {
        return TEEGNN_ERR_ALLOC;
    }
    total = data_off + block_cipher_bytes;

    if (blob_count != NULL) {
        *blob_count = h->num_blocks;
    }
    if (meta_offset != NULL) {
        *meta_offset = object_header_size;
    }
    if (data_offset != NULL) {
        *data_offset = data_off;
    }
    if (total_size != NULL) {
        *total_size = total;
    }
    if (block_payload_len != NULL) {
        *block_payload_len = block_len;
    }

    return TEEGNN_OK;
}

static EncryptedEdgeBlobMeta *blob_meta_mut(EncryptedBlockedEdgeList *enc)
{
    return (EncryptedEdgeBlobMeta *)((uint8_t *)enc + enc->meta_offset);
}

static const EncryptedEdgeBlobMeta *blob_meta_const(const EncryptedBlockedEdgeList *enc)
{
    return (const EncryptedEdgeBlobMeta *)((const uint8_t *)enc + enc->meta_offset);
}

static teegnn_status_t validate_contiguous_layout(const EncryptedBlockedEdgeList *enc)
{
    uint32_t expected_blobs = 0U;
    size_t expected_meta_offset = 0U;
    size_t expected_data_offset = 0U;
    size_t expected_total_size = 0U;
    size_t block_len = 0U;
    const EncryptedEdgeBlobMeta *meta;
    uint64_t cursor;
    uint32_t i;
    teegnn_status_t st;

    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = compute_layout_sizes(&enc->header,
                              &expected_blobs,
                              &expected_meta_offset,
                              &expected_data_offset,
                              &expected_total_size,
                              &block_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (enc->blob_count != expected_blobs ||
        enc->meta_offset != (uint64_t)expected_meta_offset ||
        enc->data_offset != (uint64_t)expected_data_offset ||
        enc->total_size != (uint64_t)expected_total_size) {
        return TEEGNN_ERR_FORMAT;
    }

    meta = blob_meta_const(enc);
    cursor = enc->data_offset;
    for (i = 0U; i < enc->blob_count; ++i) {
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
    uint32_t expected_blobs = 0U;
    size_t expected_meta_offset = 0U;
    size_t expected_data_offset = 0U;
    size_t expected_total_size = 0U;
    size_t block_len = 0U;
    const EncryptedEdgeBlobMeta *meta;
    const EncryptedEdgeBlobMeta *m;
    uint64_t expected_offset;
    teegnn_status_t st;

    if (enc == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = compute_layout_sizes(&enc->header,
                              &expected_blobs,
                              &expected_meta_offset,
                              &expected_data_offset,
                              &expected_total_size,
                              &block_len);
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

    meta = blob_meta_const(enc);
    m = &meta[blob_index];
    expected_offset = enc->data_offset + (uint64_t)blob_index * (uint64_t)block_len;

    if (m->ciphertext_offset != expected_offset ||
        m->ciphertext_len != (uint64_t)block_len ||
        m->ciphertext_offset > enc->total_size ||
        m->ciphertext_len > enc->total_size - m->ciphertext_offset) {
        return TEEGNN_ERR_FORMAT;
    }

    return TEEGNN_OK;
}

static uint32_t load_u32_slot(const uint8_t *data, uint32_t index)
{
    uint32_t value = 0U;
    TEE_MemMove(&value, data + (size_t)index * sizeof(uint32_t), sizeof(value));
    return value;
}

static double load_double_slot(const uint8_t *data, uint32_t index)
{
    double value = 0.0;
    TEE_MemMove(&value, data + (size_t)index * sizeof(double), sizeof(value));
    return value;
}

static void store_u32_slot(uint8_t *data, uint32_t index, uint32_t value)
{
    TEE_MemMove(data + (size_t)index * sizeof(uint32_t), &value, sizeof(value));
}

static void store_double_slot(uint8_t *data, uint32_t index, double value)
{
    TEE_MemMove(data + (size_t)index * sizeof(double), &value, sizeof(value));
}

teegnn_status_t encrypted_blocked_edge_list_get_blob_view(
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    EncryptedEdgeBlobView *out
) {
    const EncryptedEdgeBlobMeta *meta;
    const EncryptedEdgeBlobMeta *m;
    teegnn_status_t st;

    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    meta = blob_meta_const(enc);
    m = &meta[blob_index];

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
    EncryptedEdgeBlobMeta *meta;
    EncryptedEdgeBlobMeta *m;
    teegnn_status_t st;

    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = validate_blob_layout_at(enc, blob_index);
    if (st != TEEGNN_OK) {
        return st;
    }

    meta = blob_meta_mut(enc);
    m = &meta[blob_index];

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
    size_t off = 0U;

    if (h == NULL || aad_len == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    *aad_len = TEEGNN_EDGE_BLOCK_AAD_LEN;
    if (aad == NULL || aad_capacity < TEEGNN_EDGE_BLOCK_AAD_LEN) {
        return TEEGNN_ERR_BOUNDS;
    }

#define TEEGNN_COPY_FIELD(field) do { \
        TEE_MemMove(aad + off, &(h->field), sizeof(h->field)); \
        off += sizeof(h->field); \
    } while (0)

    TEEGNN_COPY_FIELD(graph_id);
    TEEGNN_COPY_FIELD(graph_version);
    TEEGNN_COPY_FIELD(layer_id);
    TEEGNN_COPY_FIELD(layout_version);
    TEEGNN_COPY_FIELD(n_nodes);
    TEEGNN_COPY_FIELD(m_edges);
    TEEGNN_COPY_FIELD(block_size);
    TEEGNN_COPY_FIELD(num_blocks);

#undef TEEGNN_COPY_FIELD

    TEE_MemMove(aad + off, &block_id, sizeof(block_id));
    return TEEGNN_OK;
}

static uint32_t fnv1a_mix_u32(uint32_t hash, uint32_t value)
{
    const uint8_t *bytes = (const uint8_t *)&value;
    size_t i;

    for (i = 0U; i < sizeof(value); ++i) {
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
    uint32_t context_hash = 2166136261U;

    if (h == NULL || nonce == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    context_hash = fnv1a_mix_u32(context_hash, h->graph_id);
    context_hash = fnv1a_mix_u32(context_hash, h->graph_version);
    context_hash = fnv1a_mix_u32(context_hash, h->layer_id);
    context_hash = fnv1a_mix_u32(context_hash, h->layout_version);
    context_hash = fnv1a_mix_u32(context_hash, h->n_nodes);
    context_hash = fnv1a_mix_u32(context_hash, h->m_edges);
    context_hash = fnv1a_mix_u32(context_hash, h->block_size);
    context_hash = fnv1a_mix_u32(context_hash, h->num_blocks);

    TEE_MemMove(nonce, &context_hash, sizeof(context_hash));
    TEE_MemMove(nonce + sizeof(context_hash), &block_id, sizeof(block_id));

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
    EncryptedEdgeBlobMeta *meta;
    EncryptedEdgeBlobMeta *m;
    uint8_t aad[TEEGNN_EDGE_BLOCK_AAD_LEN];
    size_t aad_len = 0U;
    teegnn_status_t st;

    if (enc == NULL || blob_index >= enc->blob_count) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    meta = blob_meta_mut(enc);
    m = &meta[blob_index];
    if (m->ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    st = derive_edge_nonce_simple(&enc->header, aad_block_id, m->nonce);
    if (st == TEEGNN_OK) {
        st = build_edge_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
    }
    if (st != TEEGNN_OK) {
        return st;
    }

    return teegnn_aes_gcm_encrypt(key,
                                  key_len,
                                  m->nonce,
                                  aad,
                                  aad_len,
                                  payload,
                                  payload_len,
                                  (uint8_t *)enc + m->ciphertext_offset,
                                  m->tag);
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
    uint8_t aad[TEEGNN_EDGE_BLOCK_AAD_LEN];
    size_t aad_len = 0U;
    teegnn_status_t st;

    st = encrypted_blocked_edge_list_get_blob_view(enc, blob_index, &view);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (view.ciphertext_len != (uint64_t)payload_len) {
        return TEEGNN_ERR_FORMAT;
    }

    st = build_edge_block_aad(&enc->header, aad_block_id, aad, sizeof(aad), &aad_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    return teegnn_aes_gcm_decrypt(key,
                                  key_len,
                                  view.nonce,
                                  aad,
                                  aad_len,
                                  view.ciphertext,
                                  payload_len,
                                  view.tag,
                                  payload);
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
    BlockedEdgeListHeader header;
    uint32_t blob_count = 0U;
    size_t meta_offset = 0U;
    size_t data_offset = 0U;
    size_t total_size = 0U;
    size_t block_payload_len = 0U;
    EncryptedBlockedEdgeList *enc = NULL;
    EncryptedEdgeBlobMeta *meta;
    uint64_t cursor;
    uint32_t block_id;
    teegnn_status_t st;

    if (out == NULL || block_size == 0U) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    *out = NULL;

    st = edge_list_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }

    TEE_MemFill(&header, 0, sizeof(header));
    header.graph_id = graph_id;
    header.graph_version = graph_version;
    header.layer_id = layer_id;
    header.layout_version = layout_version;
    header.n_nodes = A->n_nodes;
    header.m_edges = A->m_edges;
    header.block_size = block_size;
    header.num_blocks = expected_num_blocks(A->m_edges, block_size);

    st = compute_layout_sizes(&header,
                              &blob_count,
                              &meta_offset,
                              &data_offset,
                              &total_size,
                              &block_payload_len);
    if (st != TEEGNN_OK) {
        return st;
    }

    enc = (EncryptedBlockedEdgeList *)TEE_Malloc(total_size, TEE_MALLOC_FILL_ZERO);
    if (enc == NULL) {
        return TEEGNN_ERR_ALLOC;
    }

    enc->header = header;
    enc->blob_count = blob_count;
    enc->reserved = 0U;
    enc->meta_offset = (uint64_t)meta_offset;
    enc->data_offset = (uint64_t)data_offset;
    enc->total_size = (uint64_t)total_size;

    meta = blob_meta_mut(enc);
    cursor = enc->data_offset;
    for (block_id = 0U; block_id < header.num_blocks; ++block_id) {
        uint32_t blob_index = TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id);
        meta[blob_index].ciphertext_offset = cursor;
        meta[blob_index].ciphertext_len = (uint64_t)block_payload_len;
        cursor += (uint64_t)block_payload_len;
    }

    for (block_id = 0U; block_id < header.num_blocks; ++block_id) {
        uint8_t *payload;
        uint8_t *dsts;
        uint8_t *srcs;
        uint8_t *values;
        uint8_t *valid;
        uint32_t t;

        payload = (uint8_t *)TEE_Malloc(block_payload_len, TEE_MALLOC_FILL_ZERO);
        if (payload == NULL) {
            encrypted_blocked_edge_list_free(enc);
            return TEEGNN_ERR_ALLOC;
        }

        dsts = payload;
        srcs = dsts + (size_t)block_size * sizeof(uint32_t);
        values = srcs + (size_t)block_size * sizeof(uint32_t);
        valid = values + (size_t)block_size * sizeof(double);

        for (t = 0U; t < block_size; ++t) {
            uint64_t global_pos = (uint64_t)block_id * (uint64_t)block_size + (uint64_t)t;

            if (global_pos < A->m_edges) {
                const Edge *e = &A->e_ptr[global_pos];
                store_u32_slot(dsts, t, (uint32_t)e->dst);
                store_u32_slot(srcs, t, (uint32_t)e->src);
                store_double_slot(values, t, e->value);
                valid[t] = 1U;
            }
        }

        st = encrypt_blob_at_index(enc,
                                   TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id),
                                   block_id,
                                   payload,
                                   block_payload_len,
                                   key,
                                   key_len);
        TEE_Free(payload);
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
    EdgeList tmp;
    size_t block_payload_len = 0U;
    uint32_t block_id;
    teegnn_status_t st;

    if (enc == NULL || out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = validate_contiguous_layout(enc);
    if (st != TEEGNN_OK) {
        return st;
    }

    TEE_MemFill(&tmp, 0, sizeof(tmp));
    st = edge_list_alloc(&tmp, enc->header.n_nodes, enc->header.m_edges);
    if (st != TEEGNN_OK) {
        return st;
    }

    st = edge_block_payload_size(enc->header.block_size, &block_payload_len);
    if (st != TEEGNN_OK) {
        edge_list_free(&tmp);
        return st;
    }

    for (block_id = 0U; block_id < enc->header.num_blocks; ++block_id) {
        uint8_t *payload;
        const uint8_t *dsts;
        const uint8_t *srcs;
        const uint8_t *values;
        const uint8_t *valid;
        uint32_t t;

        payload = (uint8_t *)TEE_Malloc(block_payload_len, 0U);
        if (payload == NULL) {
            edge_list_free(&tmp);
            return TEEGNN_ERR_ALLOC;
        }

        st = decrypt_blob_at_index(enc,
                                   TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id),
                                   block_id,
                                   key,
                                   key_len,
                                   payload,
                                   block_payload_len);
        if (st != TEEGNN_OK) {
            TEE_Free(payload);
            edge_list_free(&tmp);
            return st;
        }

        dsts = payload;
        srcs = dsts + (size_t)enc->header.block_size * sizeof(uint32_t);
        values = srcs + (size_t)enc->header.block_size * sizeof(uint32_t);
        valid = values + (size_t)enc->header.block_size * sizeof(double);

        for (t = 0U; t < enc->header.block_size; ++t) {
            uint64_t global_pos = (uint64_t)block_id *
                                  (uint64_t)enc->header.block_size +
                                  (uint64_t)t;
            uint8_t is_valid = valid[t];

            if (is_valid > 1U) {
                TEE_Free(payload);
                edge_list_free(&tmp);
                return TEEGNN_ERR_FORMAT;
            }

            if (global_pos < enc->header.m_edges) {
                uint32_t dst;
                uint32_t src;

                if (is_valid != 1U) {
                    TEE_Free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_FORMAT;
                }

                dst = load_u32_slot(dsts, t);
                src = load_u32_slot(srcs, t);
                if (src >= enc->header.n_nodes || dst >= enc->header.n_nodes ||
                    src > (uint32_t)INT32_MAX || dst > (uint32_t)INT32_MAX) {
                    TEE_Free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_BOUNDS;
                }

                tmp.e_ptr[global_pos].src = (int32_t)src;
                tmp.e_ptr[global_pos].dst = (int32_t)dst;
                tmp.e_ptr[global_pos].value = load_double_slot(values, t);
            } else {
                uint32_t dst = load_u32_slot(dsts, t);
                uint32_t src = load_u32_slot(srcs, t);
                double value = load_double_slot(values, t);

                if (is_valid != 0U || dst != 0U || src != 0U || value != 0.0) {
                    TEE_Free(payload);
                    edge_list_free(&tmp);
                    return TEEGNN_ERR_FORMAT;
                }
            }
        }

        TEE_Free(payload);
    }

    st = edge_list_validate(&tmp);
    if (st != TEEGNN_OK) {
        edge_list_free(&tmp);
        return st;
    }

    *out = tmp;
    return TEEGNN_OK;
}

void encrypted_blocked_edge_list_free(EncryptedBlockedEdgeList *enc)
{
    TEE_Free(enc);
}
