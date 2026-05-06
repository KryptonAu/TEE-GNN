#include "csc_graph.h"

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#include <tee_internal_api.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

static int teegnn_mul_overflows_size_t(size_t a, size_t b)
{
    return b != 0U && a > SIZE_MAX / b;
}

static int teegnn_add_overflows_size_t(size_t a, size_t b)
{
    return a > SIZE_MAX - b;
}

static teegnn_status_t teegnn_array_bytes(size_t count, size_t elem_size,
                                          size_t *out)
{
    if (out == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (teegnn_mul_overflows_size_t(count, elem_size)) {
        return TEEGNN_ERR_ALLOC;
    }
    *out = count * elem_size;
    return TEEGNN_OK;
}

teegnn_status_t csc_graph_alloc(
    CSCGraph *g,
    uint32_t n_nodes,
    uint32_t nnz
)
{
    CSCGraph tmp;
    size_t col_len = 0;
    size_t col_bytes = 0;
    size_t row_bytes = 0;
    size_t val_bytes = 0;
    teegnn_status_t st;

    if (g == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (teegnn_add_overflows_size_t((size_t)n_nodes, 1U)) {
        return TEEGNN_ERR_ALLOC;
    }

    tmp.n_nodes = n_nodes;
    tmp.nnz = nnz;
    tmp.col_ptr = NULL;
    tmp.row_idx = NULL;
    tmp.values = NULL;

    col_len = (size_t)n_nodes + 1U;
    st = teegnn_array_bytes(col_len, sizeof(uint32_t), &col_bytes);
    if (st != TEEGNN_OK) {
        return st;
    }

    tmp.col_ptr = (uint32_t *)TEE_Malloc(col_bytes, TEE_MALLOC_FILL_ZERO);
    if (tmp.col_ptr == NULL) {
        return TEEGNN_ERR_ALLOC;
    }

    if (nnz > 0U) {
        st = teegnn_array_bytes((size_t)nnz, sizeof(uint32_t), &row_bytes);
        if (st != TEEGNN_OK) {
            TEE_Free(tmp.col_ptr);
            return st;
        }

        st = teegnn_array_bytes((size_t)nnz, sizeof(double), &val_bytes);
        if (st != TEEGNN_OK) {
            TEE_Free(tmp.col_ptr);
            return st;
        }

        tmp.row_idx = (uint32_t *)TEE_Malloc(row_bytes, TEE_MALLOC_FILL_ZERO);
        tmp.values = (double *)TEE_Malloc(val_bytes, TEE_MALLOC_FILL_ZERO);
        if (tmp.row_idx == NULL || tmp.values == NULL) {
            TEE_Free(tmp.col_ptr);
            TEE_Free(tmp.row_idx);
            TEE_Free(tmp.values);
            return TEEGNN_ERR_ALLOC;
        }
    }

    *g = tmp;
    return TEEGNN_OK;
}

void csc_graph_free(CSCGraph *g)
{
    if (g == NULL) {
        return;
    }

    TEE_Free(g->col_ptr);
    TEE_Free(g->row_idx);
    TEE_Free(g->values);

    g->n_nodes = 0U;
    g->nnz = 0U;
    g->col_ptr = NULL;
    g->row_idx = NULL;
    g->values = NULL;
}

teegnn_status_t csc_graph_validate(const CSCGraph *g)
{
    uint32_t col;
    uint32_t p;

    if (g == NULL || g->col_ptr == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (g->nnz > 0U && (g->row_idx == NULL || g->values == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (g->n_nodes == 0U && g->nnz != 0U) {
        return TEEGNN_ERR_FORMAT;
    }

    if (g->col_ptr[0] != 0U || g->col_ptr[g->n_nodes] != g->nnz) {
        return TEEGNN_ERR_FORMAT;
    }

    for (col = 0U; col < g->n_nodes; ++col) {
        if (g->col_ptr[col] > g->col_ptr[col + 1U] ||
            g->col_ptr[col + 1U] > g->nnz) {
            return TEEGNN_ERR_FORMAT;
        }
    }

    for (p = 0U; p < g->nnz; ++p) {
        if (g->row_idx[p] >= g->n_nodes) {
            return TEEGNN_ERR_BOUNDS;
        }
    }

    return TEEGNN_OK;
}

teegnn_status_t csc_graph_clone(
    const CSCGraph *src,
    CSCGraph *dst
)
{
    teegnn_status_t st;
    CSCGraph tmp;
    size_t col_len;
    size_t col_bytes;
    size_t row_bytes;
    size_t val_bytes;

    if (dst == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    st = csc_graph_validate(src);
    if (st != TEEGNN_OK) {
        return st;
    }

    tmp.n_nodes = 0U;
    tmp.nnz = 0U;
    tmp.col_ptr = NULL;
    tmp.row_idx = NULL;
    tmp.values = NULL;

    st = csc_graph_alloc(&tmp, src->n_nodes, src->nnz);
    if (st != TEEGNN_OK) {
        return st;
    }

    col_len = (size_t)src->n_nodes + 1U;
    st = teegnn_array_bytes(col_len, sizeof(uint32_t), &col_bytes);
    if (st != TEEGNN_OK) {
        csc_graph_free(&tmp);
        return st;
    }
    TEE_MemMove(tmp.col_ptr, src->col_ptr, col_bytes);

    if (src->nnz > 0U) {
        st = teegnn_array_bytes((size_t)src->nnz, sizeof(uint32_t), &row_bytes);
        if (st != TEEGNN_OK) {
            csc_graph_free(&tmp);
            return st;
        }

        st = teegnn_array_bytes((size_t)src->nnz, sizeof(double), &val_bytes);
        if (st != TEEGNN_OK) {
            csc_graph_free(&tmp);
            return st;
        }

        TEE_MemMove(tmp.row_idx, src->row_idx, row_bytes);
        TEE_MemMove(tmp.values, src->values, val_bytes);
    }

    *dst = tmp;
    return TEEGNN_OK;
}

teegnn_status_t csc_graph_spmm_plain(
    const CSCGraph *A,
    const double *Y,
    uint32_t feat_dim,
    double *Z
)
{
    teegnn_status_t st;
    size_t total;
    size_t total_bytes;
    uint32_t col;

    st = csc_graph_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (feat_dim > 0U && A->n_nodes > 0U && (Y == NULL || Z == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (teegnn_mul_overflows_size_t((size_t)A->n_nodes,
                                    (size_t)feat_dim)) {
        return TEEGNN_ERR_ALLOC;
    }
    total = (size_t)A->n_nodes * (size_t)feat_dim;

    if (total > 0U) {
        st = teegnn_array_bytes(total, sizeof(double), &total_bytes);
        if (st != TEEGNN_OK) {
            return st;
        }
        TEE_MemFill(Z, 0, total_bytes);
    }

    for (col = 0U; col < A->n_nodes; ++col) {
        uint32_t p;
        for (p = A->col_ptr[col]; p < A->col_ptr[col + 1U]; ++p) {
            uint32_t row = A->row_idx[p];
            double value = A->values[p];
            uint32_t f;

            for (f = 0U; f < feat_dim; ++f) {
                size_t z_idx = ((size_t)row * (size_t)feat_dim) + (size_t)f;
                size_t y_idx = ((size_t)col * (size_t)feat_dim) + (size_t)f;
                Z[z_idx] += value * Y[y_idx];
            }
        }
    }

    return TEEGNN_OK;
}
