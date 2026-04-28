#pragma once

#include "types.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace teegnn {

struct WeightedEdge {
    int row = 0;
    int col = 0;
    double value = 0.0;
};

class Graph {
public:
    Graph() = default;
    Graph(int nodes, std::size_t raw_directed_edges, std::vector<WeightedEdge> edges);

    int num_nodes() const { return num_nodes_; }
    std::size_t raw_directed_edges() const { return raw_directed_edges_; }
    const std::vector<WeightedEdge>& edges() const { return edges_; }
    const std::vector<std::vector<std::pair<int, double>>>& adjacency() const { return adjacency_; }

private:
    int num_nodes_ = 0;
    std::size_t raw_directed_edges_ = 0;
    std::vector<WeightedEdge> edges_;
    std::vector<std::vector<std::pair<int, double>>> adjacency_;
};

Graph build_normalized_graph(int num_nodes, const std::vector<std::pair<int, int>>& directed_edges);
Matrix sparse_dense_mul(const Graph& graph, const Matrix& input);
Matrix sparse_dense_mul_edges(int rows, const std::vector<WeightedEdge>& edges, const Matrix& input);

}  // namespace teegnn

