#include "graph.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace teegnn {

Graph::Graph(size_t nodes)
    : num_nodes_(nodes){
    if (num_nodes_ < 0) {
        throw std::runtime_error("graph has negative node count");
    }
    row_adj_.resize(nodes);
    col_adj_.resize(nodes);
    sqrt_degree_.resize(nodes);
}

void Graph::add_edge(uint32_t u, uint32_t v) {
    if (u < 0 || v < 0 || u >= num_nodes_ || v >= num_nodes_) {
        throw std::runtime_error("invalid edge");
    }
    row_adj_[u].push_back(v);
    col_adj_[v].push_back(u);
    sqrt_degree_[u] += 1;
    num_edges_++;
}

void Graph::sqrt() {
    for (auto &d : sqrt_degree_) {
        d = std::sqrt(d);
    }
}

Graph build_graph(size_t num_nodes, const std::vector<std::pair<int, int>>& directed_edges) {
    if (num_nodes <= 0) {
        throw std::runtime_error("dataset must contain at least one node");
    }
    Graph g(num_nodes);
    for (auto &[u, v] : directed_edges) {
        g.add_edge(u, v);
    }
    for (uint32_t i = 0; i < num_nodes; ++i) {
        g.add_edge(i, i);
    }
    g.sqrt();
    return g;
}

Matrix sparse_dense_mul(const Graph& graph, const Matrix& input) {
    if (input.rows() != graph.num_nodes()) {
        throw std::runtime_error("sparse-dense input row count does not match graph node count");
    }
    Matrix output = Matrix::Zero(graph.num_nodes(), input.cols());
    for (int row = 0; row < graph.num_nodes(); ++row) {
        for (const auto& col : graph.row_adj()[row]) {
            output.row(row).noalias() += input.row(col) / graph.sqrt_degree(row) / graph.sqrt_degree(col);
        }
    }
    return output;
}

bool mul_overflows_size_t(size_t a, size_t b) {
    return b != 0 && a > std::numeric_limits<size_t>::max() / b;
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

teegnn_status_t graph_to_csc_graph(
    const Graph& g, 
    const std::vector<uint32_t>& col_perm, 
    CSCGraph* out
) {
    if (out == nullptr) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    teegnn_status_t st;

    CSCGraph tmp{};
    st = csc_graph_alloc(&tmp, g.num_nodes(), static_cast<uint32_t>(g.num_edges()));
    if (st != TEEGNN_OK) {
        return st;
    }

    uint32_t pos = 0;
    tmp.col_ptr[0] = 0;
    for (uint32_t col = 0; col < g.num_nodes(); ++col) {
        uint32_t n_col = col_perm[col];
        for (const auto& v : g.col_adj()[n_col]) {
            tmp.row_idx[pos] = v;
            tmp.values[pos] = 1;
            ++pos;
        }
        tmp.col_ptr[col + 1] = pos;
    }

    *out = tmp;
    return TEEGNN_OK;
}

}  // namespace teegnn

