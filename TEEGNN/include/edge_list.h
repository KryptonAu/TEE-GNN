#ifndef EDGE_LIST_H
#define EDGE_LIST_H

#include <stdint.h>

#include "teegnn_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t src;
    uint32_t dst;
    double value;
} Edge;

typedef struct {
    uint32_t n_nodes;
    uint32_t m_edges;
    Edge* e_ptr;
} EdgeList;

teegnn_status_t edge_list_alloc(
    EdgeList *g,
    uint32_t n_nodes,
    uint32_t m_edges
);

void edge_list_free(EdgeList *g);

teegnn_status_t edge_list_validate(
    const EdgeList *g
);

teegnn_status_t edge_list_spmm_plain(
    const EdgeList *A,
    const double *Y,
    uint32_t feat_dim,
    double *Z
);

#ifdef __cplusplus
}
#endif

#endif
