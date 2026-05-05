#ifndef TEEGNN_CSC_GRAPH_H
#define TEEGNN_CSC_GRAPH_H

#include <stdint.h>

#include "teegnn_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n_nodes;
    uint32_t nnz;
    uint32_t *col_ptr;
    uint32_t *row_idx;
    double *values;
} CSCGraph;

teegnn_status_t csc_graph_alloc(
    CSCGraph *g,
    uint32_t n_nodes,
    uint32_t nnz
);

void csc_graph_free(CSCGraph *g);

teegnn_status_t csc_graph_validate(
    const CSCGraph *g
);

teegnn_status_t csc_graph_clone(
    const CSCGraph *src,
    CSCGraph *dst
);

teegnn_status_t csc_graph_spmm_plain(
    const CSCGraph *A,
    const double *Y,
    uint32_t feat_dim,
    double *Z
);

#ifdef __cplusplus
}
#endif

#endif
