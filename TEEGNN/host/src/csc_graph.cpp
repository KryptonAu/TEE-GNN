#include "csc_graph.h"

#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
    bool mul_overflows_size_t(size_t a, size_t b) {
        return b != 0 && a > std::numeric_limits<size_t>::max() / b;
    }
}

teegnn_status_t csc_graph_alloc(
    CSCGraph *g,
    uint32_t n_nodes,
    uint32_t nnz
) {
    if (g == nullptr) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    CSCGraph tmp{};
    tmp.n_nodes = n_nodes;
    tmp.nnz = nnz;

    const size_t col_len = static_cast<size_t>(n_nodes) + 1U;
    if (mul_overflows_size_t(col_len, sizeof(uint32_t))) {
        return TEEGNN_ERR_ALLOC;
    }
    tmp.col_ptr = static_cast<uint32_t*>(std::calloc(col_len, sizeof(uint32_t)));
    if (tmp.col_ptr == nullptr) {
        return TEEGNN_ERR_ALLOC;
    }

    if (nnz > 0) {
        if (mul_overflows_size_t(nnz, sizeof(uint32_t)) ||
            mul_overflows_size_t(nnz, sizeof(double))) {
            std::free(tmp.col_ptr);
            return TEEGNN_ERR_ALLOC;
        }
        tmp.row_idx = static_cast<uint32_t*>(std::calloc(nnz, sizeof(uint32_t)));
        tmp.values = static_cast<double*>(std::calloc(nnz, sizeof(double)));
        if (tmp.row_idx == nullptr || tmp.values == nullptr) {
            std::free(tmp.col_ptr);
            std::free(tmp.row_idx);
            std::free(tmp.values);
            return TEEGNN_ERR_ALLOC;
        }
    }

    *g = tmp;
    return TEEGNN_OK;
}

void csc_graph_free(CSCGraph *g) {
    if (g == nullptr) {
        return;
    }
    std::free(g->col_ptr);
    std::free(g->row_idx);
    std::free(g->values);
    g->n_nodes = 0;
    g->nnz = 0;
    g->col_ptr = nullptr;
    g->row_idx = nullptr;
    g->values = nullptr;
}

teegnn_status_t csc_graph_validate(
    const CSCGraph *g
) {
    if (g == nullptr || g->col_ptr == nullptr) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (g->nnz > 0 && (g->row_idx == nullptr || g->values == nullptr)) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    if (g->n_nodes == 0 && g->nnz != 0) {
        return TEEGNN_ERR_FORMAT;
    }
    if (g->col_ptr[0] != 0 || g->col_ptr[g->n_nodes] != g->nnz) {
        return TEEGNN_ERR_FORMAT;
    }
    for (uint32_t col = 0; col < g->n_nodes; ++col) {
        if (g->col_ptr[col] > g->col_ptr[col + 1] ||
            g->col_ptr[col + 1] > g->nnz) {
            return TEEGNN_ERR_FORMAT;
        }
    }
    for (uint32_t p = 0; p < g->nnz; ++p) {
        if (g->row_idx[p] >= g->n_nodes) {
            return TEEGNN_ERR_BOUNDS;
        }
    }
    return TEEGNN_OK;
}

teegnn_status_t csc_graph_clone(
    const CSCGraph *src,
    CSCGraph *dst
) {
    if (dst == nullptr) {
        return TEEGNN_ERR_INVALID_ARG;
    }
    teegnn_status_t st = csc_graph_validate(src);
    if (st != TEEGNN_OK) {
        return st;
    }

    CSCGraph tmp{};
    st = csc_graph_alloc(&tmp, src->n_nodes, src->nnz);
    if (st != TEEGNN_OK) {
        return st;
    }

    std::memcpy(
        tmp.col_ptr,
        src->col_ptr,
        (static_cast<size_t>(src->n_nodes) + 1U) * sizeof(uint32_t)
    );
    if (src->nnz > 0) {
        std::memcpy(tmp.row_idx, src->row_idx, static_cast<size_t>(src->nnz) * sizeof(uint32_t));
        std::memcpy(tmp.values, src->values, static_cast<size_t>(src->nnz) * sizeof(double));
    }
    *dst = tmp;
    return TEEGNN_OK;
}

teegnn_status_t csc_graph_spmm_plain(
    const CSCGraph *A,
    const double *Y,
    uint32_t feat_dim,
    double *Z
) {
    teegnn_status_t st = csc_graph_validate(A);
    if (st != TEEGNN_OK) {
        return st;
    }
    if (feat_dim > 0 && A->n_nodes > 0 && (Y == nullptr || Z == nullptr)) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    const size_t total = static_cast<size_t>(A->n_nodes) * static_cast<size_t>(feat_dim);
    if (feat_dim != 0 && total / feat_dim != A->n_nodes) {
        return TEEGNN_ERR_ALLOC;
    }
    if (total > 0) {
        std::memset(Z, 0, total * sizeof(double));
    }

    for (uint32_t col = 0; col < A->n_nodes; ++col) {
        for (uint32_t p = A->col_ptr[col]; p < A->col_ptr[col + 1]; ++p) {
            const uint32_t row = A->row_idx[p];
            const double value = A->values[p];
            for (uint32_t f = 0; f < feat_dim; ++f) {
                Z[static_cast<size_t>(row) * feat_dim + f] +=
                    value * Y[static_cast<size_t>(col) * feat_dim + f];
            }
        }
    }

    return TEEGNN_OK;
}
