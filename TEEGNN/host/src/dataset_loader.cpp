#include "dataset_loader.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace teegnn {
namespace {

std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir.back() == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

std::string read_text_file(const std::string& path, const std::string& kind) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open " + kind + " file: " + path);
    }

    const auto end_pos = in.tellg();
    if (end_pos == std::ifstream::pos_type(-1)) {
        throw std::runtime_error("failed to size " + kind + " file: " + path);
    }

    const std::streamoff file_size = static_cast<std::streamoff>(end_pos);
    std::string buffer(static_cast<std::size_t>(file_size), '\0');
    in.seekg(0, std::ios::beg);
    if (!buffer.empty() && !in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()))) {
        throw std::runtime_error("failed to read " + kind + " file: " + path);
    }
    return buffer;
}

bool is_space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

const char* skip_space(const char* current, const char* end) {
    while (current < end && is_space(*current)) {
        ++current;
    }
    return current;
}

std::size_t count_tokens(const char* begin, const char* end) {
    std::size_t count = 0;
    const char* current = begin;
    while (true) {
        current = skip_space(current, end);
        if (current == end) {
            return count;
        }
        ++count;
        while (current < end && !is_space(*current)) {
            ++current;
        }
    }
}

std::size_t count_tokens(const std::string& buffer) {
    return count_tokens(buffer.data(), buffer.data() + buffer.size());
}

template <typename Fn>
void for_each_line(const std::string& buffer, Fn fn) {
    const char* const file_end = buffer.data() + buffer.size();
    const char* current = buffer.data();
    while (current < file_end) {
        const char* const line_begin = current;
        while (current < file_end && *current != '\n') {
            ++current;
        }

        const char* line_end = current;
        if (line_end > line_begin && *(line_end - 1) == '\r') {
            --line_end;
        }
        fn(line_begin, line_end);

        if (current < file_end) {
            ++current;
        }
    }
}

struct MatrixShape {
    std::size_t rows = 0;
    std::size_t cols = 0;
};

MatrixShape read_matrix_shape(const std::string& buffer, const std::string& path) {
    MatrixShape shape;
    for_each_line(buffer, [&](const char* line_begin, const char* line_end) {
        const std::size_t cols = count_tokens(line_begin, line_end);
        if (cols == 0) {
            return;
        }
        if (shape.cols == 0) {
            shape.cols = cols;
        } else if (cols != shape.cols) {
            throw std::runtime_error("ragged matrix file: " + path);
        }
        ++shape.rows;
    });

    if (shape.rows == 0 || shape.cols == 0) {
        throw std::runtime_error("empty matrix file: " + path);
    }
    return shape;
}

int checked_matrix_dim(std::size_t dim, const std::string& path) {
    if (dim > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("matrix dimension too large in: " + path);
    }
    return static_cast<int>(dim);
}

double parse_double_token(const char*& current, const char* end, const std::string& path) {
    current = skip_space(current, end);
    if (current == end) {
        throw std::runtime_error("ragged matrix file: " + path);
    }

    char* next = nullptr;
    const double value = std::strtod(current, &next);
    if (next == current || next > end || (next < end && !is_space(*next))) {
        throw std::runtime_error("invalid floating-point value in matrix file: " + path);
    }

    current = next;
    return value;
}

int parse_int_token(const char*& current, const char* end, const std::string& path, const std::string& kind) {
    current = skip_space(current, end);
    if (current == end) {
        throw std::runtime_error("missing integer value in " + kind + " file: " + path);
    }

    char* next = nullptr;
    errno = 0;
    const long value = std::strtol(current, &next, 10);
    if (errno == ERANGE || next == current || next > end || (next < end && !is_space(*next))) {
        throw std::runtime_error("invalid integer value in " + kind + " file: " + path);
    }
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("integer value out of range in " + kind + " file: " + path);
    }

    current = next;
    return static_cast<int>(value);
}

Matrix read_matrix(const std::string& path) {
    const std::string buffer = read_text_file(path, "matrix");
    const MatrixShape shape = read_matrix_shape(buffer, path);
    Matrix matrix(checked_matrix_dim(shape.rows, path), checked_matrix_dim(shape.cols, path));

    double* output = matrix.data();
    for_each_line(buffer, [&](const char* line_begin, const char* line_end) {
        if (skip_space(line_begin, line_end) == line_end) {
            return;
        }

        const char* current = line_begin;
        for (std::size_t col = 0; col < shape.cols; ++col) {
            *output++ = parse_double_token(current, line_end, path);
        }
        if (skip_space(current, line_end) != line_end) {
            throw std::runtime_error("ragged matrix file: " + path);
        }
    });
    return matrix;
}

Matrix read_matrix_transposed(const std::string& path) {
    const std::string buffer = read_text_file(path, "matrix");
    const MatrixShape shape = read_matrix_shape(buffer, path);
    const int rows = checked_matrix_dim(shape.cols, path);
    const int cols = checked_matrix_dim(shape.rows, path);
    Matrix matrix(rows, cols);

    std::size_t row = 0;
    for_each_line(buffer, [&](const char* line_begin, const char* line_end) {
        if (skip_space(line_begin, line_end) == line_end) {
            return;
        }

        const char* current = line_begin;
        const int row_index = checked_matrix_dim(row, path);
        for (std::size_t col = 0; col < shape.cols; ++col) {
            matrix(static_cast<int>(col), row_index) = parse_double_token(current, line_end, path);
        }
        if (skip_space(current, line_end) != line_end) {
            throw std::runtime_error("ragged matrix file: " + path);
        }
        ++row;
    });
    return matrix;
}

IntVector read_labels(const std::string& path) {
    const std::string buffer = read_text_file(path, "labels");
    IntVector labels;
    labels.reserve(count_tokens(buffer));

    const char* current = buffer.data();
    const char* const end = current + buffer.size();
    while (true) {
        current = skip_space(current, end);
        if (current == end) {
            break;
        }
        labels.push_back(parse_int_token(current, end, path, "labels"));
    }

    if (labels.empty()) {
        throw std::runtime_error("empty labels file: " + path);
    }
    return labels;
}

std::vector<std::pair<int, int>> read_edges(const std::string& path, int& max_node) {
    const std::string buffer = read_text_file(path, "edges");
    std::vector<std::pair<int, int>> edges;
    edges.reserve(count_tokens(buffer) / 2);

    const char* current = buffer.data();
    const char* const end = current + buffer.size();
    max_node = -1;
    while (true) {
        current = skip_space(current, end);
        if (current == end) {
            break;
        }

        const int u = parse_int_token(current, end, path, "edges");
        current = skip_space(current, end);
        if (current == end) {
            throw std::runtime_error("missing edge endpoint in: " + path);
        }
        const int v = parse_int_token(current, end, path, "edges");
        if (u < 0 || v < 0) {
            throw std::runtime_error("negative edge endpoint in: " + path);
        }
        max_node = std::max(max_node, std::max(u, v));
        edges.emplace_back(u, v);
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
    dataset.w1 = read_matrix_transposed(path_join(dataset_dir, "w1.txt"));
    dataset.w2 = read_matrix_transposed(path_join(dataset_dir, "w2.txt"));

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

    dataset.graph = build_graph(num_nodes, directed_edges);
    return dataset;
}

}  // namespace teegnn
