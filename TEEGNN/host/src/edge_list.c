#include "edge_list.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int mul_overflows_size_t(size_t a, size_t b) {
    return b != 0 && a > SIZE_MAX / b;
}

teegnn_status_t edge_list_alloc(
    EdgeList *g,
    uint32_t n_nodes,
    uint32_t m_edges
) {
    if (g == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    EdgeList tmp;
    memset(&tmp, 0, sizeof(tmp));
    tmp.n_nodes = n_nodes;
    tmp.m_edges = m_edges;

    if (m_edges > 0) {
        if (mul_overflows_size_t((size_t)m_edges, sizeof(Edge))) {
            return TEEGNN_ERR_ALLOC;
        }
        tmp.e_ptr = (Edge *)calloc((size_t)m_edges, sizeof(Edge));
        if (tmp.e_ptr == NULL) {
            return TEEGNN_ERR_ALLOC;
        }
    }

    *g = tmp;
    return TEEGNN_OK;
}

void edge_list_free(
    EdgeList *g
) {
    if (g == NULL) {
        return;
    }
    free(g->e_ptr);
    g->n_nodes = 0;
    g->m_edges = 0;
    g->e_ptr = NULL;
}

teegnn_status_t edge_list_validate(
    const EdgeList *g
) {
    if (g == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (g->m_edges > 0 && g->e_ptr == NULL) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (g->n_nodes == 0 && g->m_edges != 0) {
        return TEEGNN_ERR_FORMAT;
    }

    for (uint32_t i = 0; i < g->m_edges; ++i) {
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
    teegnn_status_t st = edge_list_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (feat_dim > 0 && A->n_nodes > 0 && (Y == NULL || Z == NULL)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    const size_t total = (size_t)A->n_nodes * (size_t)feat_dim;
    if (feat_dim != 0 && total / feat_dim != A->n_nodes) {
        return TEEGNN_ERR_ALLOC;
    }
    if (mul_overflows_size_t(total, sizeof(double))) {
        return TEEGNN_ERR_ALLOC;
    }
    if (total > 0) {
        memset(Z, 0, total * sizeof(double));
    }

    for (uint32_t i = 0; i < A->m_edges; ++i) {
        const Edge *e = &A->e_ptr[i];
        const uint32_t src = (uint32_t)e->src;
        const uint32_t dst = (uint32_t)e->dst;
        for (uint32_t f = 0; f < feat_dim; ++f) {
            Z[(size_t)dst * feat_dim + f] +=
                e->value * Y[(size_t)src * feat_dim + f];
        }
    }

    return TEEGNN_OK;
}
