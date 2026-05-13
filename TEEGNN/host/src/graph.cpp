#include "graph.hpp"
#include "edge_list.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

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
            tmp.values[pos] = 1.0 / g.sqrt_degree(n_col) / g.sqrt_degree(v);
            ++pos;
        }
        tmp.col_ptr[col + 1] = pos;
    }

    *out = tmp;
    return TEEGNN_OK;
}

teegnn_status_t graph_to_edge_list(
    const Graph& g, 
    const std::vector<uint32_t>& col_perm, 
    const std::vector<uint32_t>& next_row_perm,
    size_t row_block_size,
    EdgeList* out
) {
    if (out == nullptr) {
        return TEEGNN_ERR_INVALID_ARG;
    }

    teegnn_status_t st;

    EdgeList tmp{};
    st = edge_list_alloc(&tmp, g.num_nodes(), g.num_edges());
    if (st != TEEGNN_OK) {
        return st;
    }

    std::vector<uint32_t> next_row_perm_inv(next_row_perm.size());
    for (uint32_t i = 0; i < next_row_perm.size(); ++i) {
        next_row_perm_inv[next_row_perm[i]] = i;
    }

    size_t blocks = (g.num_nodes() + row_block_size - 1U) / row_block_size;
    std::vector<std::vector<Edge>> row_blocks(blocks);
    for (uint32_t col = 0; col < g.num_nodes(); ++col) {
        uint32_t n_col = col_perm[col];
        for (const auto& v : g.col_adj()[n_col]) {
            uint32_t n_row = next_row_perm_inv[v];
            double value = 1.0 / g.sqrt_degree(n_col) / g.sqrt_degree(v);
            row_blocks[n_row / row_block_size].emplace_back(Edge{col, n_row, value});
        }
    }
    uint32_t pos = 0;
    for (const auto& block : row_blocks) {
        for (const auto& e : block) {
            tmp.e_ptr[pos] = e;
            pos++;
        }
    }

    *out = tmp;
    return TEEGNN_OK;
}

}  // namespace teegnn

