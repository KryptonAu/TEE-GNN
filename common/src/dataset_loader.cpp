#include "dataset_loader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace teegnn {
namespace {

std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir.back() == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

Matrix read_matrix(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open matrix file: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    std::size_t cols = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::vector<double> row;
        double value = 0.0;
        while (iss >> value) {
            row.push_back(value);
        }
        if (row.empty()) {
            continue;
        }
        if (cols == 0) {
            cols = row.size();
        } else if (row.size() != cols) {
            throw std::runtime_error("ragged matrix file: " + path);
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty() || cols == 0) {
        throw std::runtime_error("empty matrix file: " + path);
    }

    Matrix matrix(rows.size(), static_cast<int>(cols));
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
    }
    return matrix;
}

IntVector read_labels(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open labels file: " + path);
    }
    IntVector labels;
    int label = 0;
    while (in >> label) {
        labels.push_back(label);
    }
    if (labels.empty()) {
        throw std::runtime_error("empty labels file: " + path);
    }
    return labels;
}

std::vector<std::pair<int, int>> read_edges(const std::string& path, int& max_node) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open edges file: " + path);
    }
    std::vector<std::pair<int, int>> edges;
    int u = 0;
    int v = 0;
    max_node = -1;
    while (in >> u >> v) {
        if (u < 0 || v < 0) {
            throw std::runtime_error("negative edge endpoint in: " + path);
        }
        max_node = std::max(max_node, std::max(u, v));
        edges.push_back({u, v});
    }
    if (edges.empty()) {
        throw std::runtime_error("empty edges file: " + path);
    }
    return edges;
}

}  // namespace

Dataset load_dataset(const std::string& dataset_dir) {
    Dataset dataset;
    dataset.features = read_matrix(path_join(dataset_dir, "features.txt"));
    dataset.labels = read_labels(path_join(dataset_dir, "labels.txt"));
    dataset.w1 = read_matrix(path_join(dataset_dir, "w1.txt")).transpose();
    dataset.w2 = read_matrix(path_join(dataset_dir, "w2.txt")).transpose();

    int max_node = -1;
    auto directed_edges = read_edges(path_join(dataset_dir, "edges.txt"), max_node);
    const int num_nodes = std::max<int>(dataset.features.rows(), max_node + 1);
    if (dataset.labels.size() != static_cast<std::size_t>(num_nodes)) {
        throw std::runtime_error("labels count does not match node count");
    }
    if (dataset.features.rows() != num_nodes) {
        throw std::runtime_error("features row count does not match node count");
    }
    if (dataset.features.cols() != dataset.w1.rows()) {
        throw std::runtime_error("features columns do not match transposed w1 rows");
    }
    if (dataset.w1.cols() != dataset.w2.rows()) {
        throw std::runtime_error("hidden dimension from w1 does not match transposed w2 rows");
    }

    dataset.graph = build_normalized_graph(num_nodes, directed_edges);
    return dataset;
}

}  // namespace teegnn

