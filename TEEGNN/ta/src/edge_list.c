#include "edge_list.h"

#include <stddef.h>
#include <stdint.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)
#endif

static int teegnn_mul_overflows_size_t(size_t a, size_t b)
{
    return b != 0U && a > SIZE_MAX / b;
}

static teegnn_status_t teegnn_array_bytes(size_t count, size_t elem_size, size_t *out)
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

teegnn_status_t edge_list_alloc(
    EdgeList *g,
    uint32_t n_nodes,
    uint32_t m_edges
) {
    EdgeList tmp;
    size_t edge_bytes = 0U;
    teegnn_status_t st;

    if (g == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    tmp.n_nodes = n_nodes;
    tmp.m_edges = m_edges;
    tmp.e_ptr = NULL;

    if (m_edges > 0U) {
        st = teegnn_array_bytes((size_t)m_edges, sizeof(Edge), &edge_bytes);
        if (st != TEEGNN_OK) {
            return st;
        }

        tmp.e_ptr = (Edge *)TEE_Malloc(edge_bytes, TEE_MALLOC_FILL_ZERO);
        if (tmp.e_ptr == NULL) {
            return TEEGNN_ERR_ALLOC;
        }
    }

    *g = tmp;
    return TEEGNN_OK;
}

void edge_list_free(EdgeList *g)
{
    if (g == NULL) {
        return;
    }

    TEE_Free(g->e_ptr);

    g->n_nodes = 0U;
    g->m_edges = 0U;
    g->e_ptr = NULL;
}

teegnn_status_t edge_list_validate(const EdgeList *g)
{
    uint32_t i;

    if (g == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (g->m_edges > 0U && g->e_ptr == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    if (g->n_nodes == 0U && g->m_edges != 0U) {
        return TEEGNN_ERR_FORMAT;
    }

    for (i = 0U; i < g->m_edges; ++i) {
        const Edge *e = &g->e_ptr[i];

        if (e->src < 0 || e->dst < 0 ||
            (uint32_t)e->src >= g->n_nodes ||
            (uint32_t)e->dst >= g->n_nodes) {
            return TEEGNN_ERR_BOUNDS;
        }
    }

    return TEEGNN_OK;
}

teegnn_status_t edge_list_spmm_plain(
    const EdgeList *A,
    const double *Y,
    uint32_t feat_dim,
    double *Z
) {
    teegnn_status_t st;
    size_t total;
    size_t total_bytes;
    uint32_t i;

    st = edge_list_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }

    if (feat_dim > 0U && A->n_nodes > 0U && (Y == NULL || Z == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    total = (size_t)A->n_nodes * (size_t)feat_dim;
    if (feat_dim != 0U && total / (size_t)feat_dim != (size_t)A->n_nodes) {
        return TEEGNN_ERR_ALLOC;
    }

    if (total > 0U) {
        st = teegnn_array_bytes(total, sizeof(double), &total_bytes);
        if (st != TEEGNN_OK) {
            return st;
        }
        TEE_MemFill(Z, 0, total_bytes);
    }

    for (i = 0U; i < A->m_edges; ++i) {
        const Edge *e = &A->e_ptr[i];
        uint32_t src = (uint32_t)e->src;
        uint32_t dst = (uint32_t)e->dst;
        uint32_t f;

        for (f = 0U; f < feat_dim; ++f) {
            size_t z_idx = ((size_t)dst * (size_t)feat_dim) + (size_t)f;
            size_t y_idx = ((size_t)src * (size_t)feat_dim) + (size_t)f;

            Z[z_idx] += e->value * Y[y_idx];
        }
    }

    return TEEGNN_OK;
}
