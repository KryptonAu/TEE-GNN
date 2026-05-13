#ifndef BLOCKED_EDGE_LIST_H
#define BLOCKED_EDGE_LIST_H

#include <stddef.h>
#include <stdint.h>

#include "edge_list.h"
#include "crypto.h"
#include "teegnn_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t graph_id;
    uint32_t graph_version;
    uint32_t layer_id;
    uint32_t layout_version;
    uint32_t n_nodes;
    uint32_t m_edges;
    uint32_t block_size;
    uint32_t num_blocks;
} BlockedEdgeListHeader;

typedef struct {
    uint64_t ciphertext_offset;
    uint64_t ciphertext_len;
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN];
    uint8_t tag[TEEGNN_GCM_TAG_LEN];
} EncryptedEdgeBlobMeta;

typedef struct {
    const uint8_t *nonce;
    const uint8_t *tag;
    uint64_t ciphertext_len;
    const uint8_t *ciphertext;
} EncryptedEdgeBlobView;

typedef struct {
    uint8_t *nonce;
    uint8_t *tag;
    uint64_t ciphertext_len;
    uint8_t *ciphertext;
} EncryptedEdgeBlobMutableView;

typedef struct {
    BlockedEdgeListHeader header;
    uint32_t blob_count;
    uint32_t reserved;
    uint64_t meta_offset;
    uint64_t data_offset;
    uint64_t total_size;
    uint8_t data[1];
} EncryptedBlockedEdgeList;

#define TEEGNN_EDGE_BLOB_INDEX_BLOCK(block_id) ((uint32_t)(block_id))

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
);

teegnn_status_t blocked_edge_list_decrypt_full(
    const EncryptedBlockedEdgeList *enc,
    const uint8_t *key,
    size_t key_len,
    EdgeList *out
);

void encrypted_blocked_edge_list_free(
    EncryptedBlockedEdgeList *enc
);

teegnn_status_t encrypted_blocked_edge_list_get_blob_view(
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    EncryptedEdgeBlobView *out
);

teegnn_status_t encrypted_blocked_edge_list_get_blob_mutable_view(
    EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    EncryptedEdgeBlobMutableView *out
);

teegnn_status_t build_edge_block_aad(
    const BlockedEdgeListHeader *h,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
);

teegnn_status_t derive_edge_nonce_simple(
    const BlockedEdgeListHeader *h,
    uint64_t block_id,
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN]
);

#ifdef __cplusplus
}
#endif

#endif
