#include "graph.hpp"

#include <cmath>
#include <stdexcept>

namespace teegnn {

Graph::Graph(int nodes, std::size_t raw_directed_edges, std::vector<WeightedEdge> edges)
    : num_nodes_(nodes), raw_directed_edges_(raw_directed_edges), edges_(std::move(edges)) {
    if (num_nodes_ < 0) {
        throw std::runtime_error("graph has negative node count");
    }
    adjacency_.assign(num_nodes_, {});
    for (const auto& edge : edges_) {
        if (edge.row < 0 || edge.row >= num_nodes_ || edge.col < 0 || edge.col >= num_nodes_) {
            throw std::runtime_error("normalized graph edge index out of range");
        }
        adjacency_[edge.row].push_back({edge.col, edge.value});
    }
}

Graph build_normalized_graph(int num_nodes, const std::vector<std::pair<int, int>>& directed_edges) {
    if (num_nodes <= 0) {
        throw std::runtime_error("dataset must contain at least one node");
    }

    std::vector<int> degree(num_nodes, 1);
    for (const auto& [u, v] : directed_edges) {
        if (u < 0 || u >= num_nodes || v < 0 || v >= num_nodes) {
            throw std::runtime_error("edge endpoint out of range");
        }
        degree[u] += 1;
    }

    std::vector<WeightedEdge> normalized;
    normalized.reserve(directed_edges.size() + static_cast<std::size_t>(num_nodes));
    for (const auto& [u, v] : directed_edges) {
        const double weight = 1.0 / std::sqrt(static_cast<double>(degree[u]) * degree[v]);
        normalized.push_back({u, v, weight});
    }
    for (int i = 0; i < num_nodes; ++i) {
        normalized.push_back({i, i, 1.0 / static_cast<double>(degree[i])});
    }
    return Graph(num_nodes, directed_edges.size(), std::move(normalized));
}

Matrix sparse_dense_mul(const Graph& graph, const Matrix& input) {
    if (input.rows() != graph.num_nodes()) {
        throw std::runtime_error("sparse-dense input row count does not match graph node count");
    }
    Matrix output = Matrix::Zero(graph.num_nodes(), input.cols());
    for (int row = 0; row < graph.num_nodes(); ++row) {
        for (const auto& [col, value] : graph.adjacency()[row]) {
            output.row(row).noalias() += value * input.row(col);
        }
    }
    return output;
}

Matrix sparse_dense_mul_edges(int rows, const std::vector<WeightedEdge>& edges, const Matrix& input) {
    Matrix output = Matrix::Zero(rows, input.cols());
    for (const auto& edge : edges) {
        if (edge.row < 0 || edge.row >= rows || edge.col < 0 || edge.col >= input.rows()) {
            throw std::runtime_error("masked sparse edge index out of range");
        }
        output.row(edge.row).noalias() += edge.value * input.row(edge.col);
    }
    return output;
}

}  // namespace teegnn

