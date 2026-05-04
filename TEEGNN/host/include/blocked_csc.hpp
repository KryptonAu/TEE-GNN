#pragma once

#include <cstddef>
#include <cstdint>

#include "crypto.h"
#include "teegnn_error.h"
#include "graph.hpp"

namespace teegnn {

struct BlockedCSCHeader {
    uint32_t graph_id;
    uint32_t graph_version;
    uint32_t layer_id;
    uint32_t layout_version;
    uint32_t n_nodes;
    uint32_t nnz;
    uint32_t block_size;
    uint32_t num_blocks;
};

struct EncryptedBlobMeta {
    uint64_t ciphertext_offset;
    uint64_t ciphertext_len;
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN];
    uint8_t tag[TEEGNN_GCM_TAG_LEN];
};

struct EncryptedBlobView {
    const uint8_t *nonce;
    const uint8_t *tag;
    uint64_t ciphertext_len;
    const uint8_t *ciphertext;
};

struct EncryptedBlobMutableView {
    uint8_t *nonce;
    uint8_t *tag;
    uint64_t ciphertext_len;
    uint8_t *ciphertext;
};

struct EncryptedBlockedCSC {
    BlockedCSCHeader header;
    uint32_t blob_count;
    uint32_t reserved;
    uint64_t meta_offset;
    uint64_t data_offset;
    uint64_t total_size;
    uint8_t data[1];
};

#define TEEGNN_BLOB_INDEX_COL_PTR 0U
#define TEEGNN_BLOB_INDEX_ROW_BLOCK(block_id) ((uint32_t)((block_id) + 1U))

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
);

teegnn_status_t blocked_csc_decrypt_full(
    const EncryptedBlockedCSC *enc,
    const uint8_t *key,
    size_t key_len,
    CSCGraph *out
);

void encrypted_blocked_csc_free(
    EncryptedBlockedCSC *enc
);

teegnn_status_t encrypted_blocked_csc_get_blob_view(
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    EncryptedBlobView *out
);

teegnn_status_t encrypted_blocked_csc_get_blob_mutable_view(
    EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    EncryptedBlobMutableView *out
);

teegnn_status_t build_block_aad(
    const BlockedCSCHeader *h,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
);

teegnn_status_t derive_nonce_simple(
    const BlockedCSCHeader *h,
    uint64_t block_id,
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN]
);

}  // namespace teegnn

