#pragma once

#include "types.hpp"
#include "teegnn_error.h"
#include "csc_graph.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace teegnn {

class Graph {
public:
    Graph() = default;
    Graph(size_t nodes);

    size_t num_nodes() const { return num_nodes_; }
    size_t num_edges() const { return num_edges_; }
    void add_edge(uint32_t u, uint32_t v);
    void sqrt();
    double sqrt_degree(uint32_t u) const { return sqrt_degree_[u];}
    const std::vector<std::vector<uint32_t>>& row_adj() const { return row_adj_; }
    const std::vector<std::vector<uint32_t>>& col_adj() const { return col_adj_; }

private:
    size_t num_nodes_ = 0;
    size_t num_edges_ = 0;
    std::vector<std::vector<uint32_t>> row_adj_;
    std::vector<std::vector<uint32_t>> col_adj_;
    std::vector<double> sqrt_degree_;
};

Graph build_graph(size_t num_nodes, const std::vector<std::pair<int, int>>& directed_edges);
Matrix sparse_dense_mul(const Graph& graph, const Matrix& input);

teegnn_status_t graph_to_csc_graph(
    const Graph& g, 
    const std::vector<uint32_t>& col_perm, 
    CSCGraph* out
);

}  // namespace teegnn

