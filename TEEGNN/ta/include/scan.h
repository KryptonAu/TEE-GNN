#ifndef TEEGNN_SCAN_H
#define TEEGNN_SCAN_H

#include <stddef.h>
#include <stdint.h>

#include "blocked_edge_list.h"
#include "blocked_csc.h"
#include "teegnn_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TEEGNN_COL_PTR_BLOCK_ID UINT64_MAX
#define TEEGNN_BLOCK_AAD_LEN (8U * sizeof(uint32_t) + sizeof(uint64_t))
#define TEEGNN_EDGE_BLOCK_AAD_LEN (8U * sizeof(uint32_t) + sizeof(uint64_t))
#define TEEGNN_MATRIX_BLOCK_AAD_LEN (4U * sizeof(uint32_t) + sizeof(uint64_t))


int mul_overflows_size_t(size_t a, size_t b);

uint32_t expected_num_blocks(uint32_t nnz, uint32_t block_size);

teegnn_status_t validate_header_for_scan(const BlockedCSCHeader *h);

teegnn_status_t col_ptr_payload_size(uint32_t n_nodes, size_t *out);

teegnn_status_t row_block_payload_size(uint32_t block_size, size_t *out);

teegnn_status_t decrypt_blob_for_scan(
    const BlockedCSCHeader *h,
    const EncryptedBlockedCSC *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *payload,
    size_t payload_len
);

uint32_t load_u32_slot(const uint8_t *rows, uint32_t index);

double load_double_slot(const uint8_t *values, uint32_t index);

teegnn_status_t validate_col_ptr(
    const uint32_t *col_ptr,
    uint32_t n_nodes,
    uint32_t nnz
);

uint32_t expected_edge_num_blocks(uint32_t m_edges, uint32_t block_size);

teegnn_status_t validate_edge_header_for_scan(const BlockedEdgeListHeader *h);

teegnn_status_t edge_block_payload_size(uint32_t block_size, size_t *out);

teegnn_status_t validate_edge_object_layout_for_scan(
    const EncryptedBlockedEdgeList *enc
);

teegnn_status_t decrypt_edge_blob_for_scan(
    const BlockedEdgeListHeader *h,
    const EncryptedBlockedEdgeList *enc,
    uint32_t blob_index,
    uint64_t aad_block_id,
    const uint8_t *key,
    size_t key_len,
    uint8_t *payload,
    size_t payload_len
);

teegnn_status_t matrix_block_layout(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t *blocks,
    size_t *payload_len,
    size_t *block_stride
);

teegnn_status_t build_matrix_block_aad(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks,
    uint64_t block_id,
    uint8_t *aad,
    size_t aad_capacity,
    size_t *aad_len
);

teegnn_status_t derive_matrix_nonce_simple(
    uint32_t row_block_size,
    uint32_t rows,
    uint32_t cols,
    uint32_t blocks,
    uint64_t block_id,
    uint8_t nonce[TEEGNN_GCM_NONCE_LEN]
);

#ifdef __cplusplus
}
#endif

#endif
