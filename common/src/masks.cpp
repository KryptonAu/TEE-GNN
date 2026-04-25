#include "masks.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace teegnn {
namespace {

void require_square_dim(int expected, int actual, const char* what) {
    if (expected != actual) {
        throw std::runtime_error(std::string(what) + " dimension mismatch");
    }
}

std::uint64_t edge_key(int row, int col, int n) {
    return static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(col);
}

}  // namespace

ScaledPermutation::ScaledPermutation(std::vector<int> permutation, std::vector<double> scale)
    : permutation_(std::move(permutation)), scale_(std::move(scale)) {
    if (permutation_.size() != scale_.size()) {
        throw std::runtime_error("scaled permutation size mismatch");
    }
    inverse_permutation_.assign(permutation_.size(), 0);
    std::vector<int> seen(permutation_.size(), 0);
    for (int i = 0; i < dim(); ++i) {
        const int mapped = permutation_[static_cast<std::size_t>(i)];
        if (mapped < 0 || mapped >= dim() || seen[static_cast<std::size_t>(mapped)] != 0) {
            throw std::runtime_error("invalid permutation");
        }
        if (std::abs(scale_[static_cast<std::size_t>(i)]) < 1e-12) {
            throw std::runtime_error("scaled permutation contains zero scale");
        }
        seen[static_cast<std::size_t>(mapped)] = 1;
        inverse_permutation_[static_cast<std::size_t>(mapped)] = i;
    }
}

ScaledPermutation ScaledPermutation::random(int dim, RandomEngine& rng) {
    std::vector<int> permutation(static_cast<std::size_t>(dim));
    std::vector<double> scale(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        permutation[static_cast<std::size_t>(i)] = i;
        scale[static_cast<std::size_t>(i)] = rng.nonzero_scale();
    }
    for (int i = dim - 1; i > 0; --i) {
        const int j = static_cast<int>(rng.uniform_index(static_cast<std::uint64_t>(i + 1)));
        std::swap(permutation[static_cast<std::size_t>(i)], permutation[static_cast<std::size_t>(j)]);
    }
    return ScaledPermutation(std::move(permutation), std::move(scale));
}

Matrix ScaledPermutation::apply_left(const Matrix& x) const {
    require_square_dim(dim(), x.rows(), "left P");
    Matrix out(x.rows(), x.cols());
    for (int i = 0; i < dim(); ++i) {
        out.row(i) = scale_[static_cast<std::size_t>(i)] * x.row(permutation_[static_cast<std::size_t>(i)]);
    }
    return out;
}

Matrix ScaledPermutation::apply_left_inv(const Matrix& x) const {
    require_square_dim(dim(), x.rows(), "left P inverse");
    Matrix out(x.rows(), x.cols());
    for (int i = 0; i < dim(); ++i) {
        out.row(permutation_[static_cast<std::size_t>(i)]) =
            x.row(i) / scale_[static_cast<std::size_t>(i)];
    }
    return out;
}

Matrix ScaledPermutation::apply_right(const Matrix& x) const {
    require_square_dim(dim(), x.cols(), "right P");
    Matrix out(x.rows(), x.cols());
    for (int i = 0; i < dim(); ++i) {
        out.col(permutation_[static_cast<std::size_t>(i)]) =
            scale_[static_cast<std::size_t>(i)] * x.col(i);
    }
    return out;
}

Matrix ScaledPermutation::apply_right_inv(const Matrix& x) const {
    require_square_dim(dim(), x.cols(), "right P inverse");
    Matrix out(x.rows(), x.cols());
    for (int i = 0; i < dim(); ++i) {
        out.col(i) = x.col(permutation_[static_cast<std::size_t>(i)]) /
                     scale_[static_cast<std::size_t>(i)];
    }
    return out;
}

Matrix LowRankMask::materialize() const {
    return u * v.transpose();
}

Matrix LowRankMask::ahat_times_mask(const Graph& graph) const {
    return sparse_dense_mul(graph, u) * v.transpose();
}

LowRankMask make_low_rank_mask(int rows, int cols, int rank, RandomEngine& rng, double amplitude) {
    if (rank <= 0) {
        return {Matrix::Zero(rows, 0), Matrix::Zero(cols, 0)};
    }
    return {rng.normal_like(rows, rank, amplitude), rng.normal_like(cols, rank, amplitude)};
}

SDIMMask::SDIMMask(ScaledPermutation p, Vector u, Vector v)
    : p_(std::move(p)), u_(std::move(u)), v_(std::move(v)) {
    if (u_.size() != p_.dim() || v_.size() != p_.dim()) {
        throw std::runtime_error("SDIM vector dimension mismatch");
    }
    Matrix u_mat(u_.rows(), 1);
    u_mat.col(0) = u_;
    const Matrix pinv_u = p_.apply_left_inv(u_mat);
    denominator_ = 1.0 + v_.dot(pinv_u.col(0));
    if (std::abs(denominator_) < 1e-8) {
        throw std::runtime_error("singular SDIM mask");
    }
}

SDIMMask SDIMMask::random(int dim, RandomEngine& rng) {
    for (int attempt = 0; attempt < 32; ++attempt) {
        ScaledPermutation p = ScaledPermutation::random(dim, rng);
        Vector u(dim);
        Vector v(dim);
        for (int i = 0; i < dim; ++i) {
            u(i) = rng.uniform(-0.02, 0.02);
            v(i) = rng.uniform(-0.02, 0.02);
        }
        try {
            return SDIMMask(std::move(p), std::move(u), std::move(v));
        } catch (const std::runtime_error&) {
        }
    }
    throw std::runtime_error("failed to generate non-singular SDIM mask");
}

Matrix SDIMMask::apply_left(const Matrix& x) const {
    require_square_dim(dim(), x.rows(), "left SDIM");
    Matrix out = p_.apply_left(x);
    out.noalias() += u_ * (v_.transpose() * x);
    return out;
}

Matrix SDIMMask::apply_left_inv(const Matrix& x) const {
    require_square_dim(dim(), x.rows(), "left SDIM inverse");
    Matrix z = p_.apply_left_inv(x);
    Matrix u_mat(u_.rows(), 1);
    u_mat.col(0) = u_;
    const Matrix pinv_u = p_.apply_left_inv(u_mat);
    const Matrix correction = pinv_u.col(0) * ((v_.transpose() * z) / denominator_);
    z -= correction;
    return z;
}

Matrix SDIMMask::apply_right(const Matrix& x) const {
    require_square_dim(dim(), x.cols(), "right SDIM");
    Matrix out = p_.apply_right(x);
    out.noalias() += (x * u_) * v_.transpose();
    return out;
}

Matrix SDIMMask::apply_right_inv(const Matrix& x) const {
    require_square_dim(dim(), x.cols(), "right SDIM inverse");
    Matrix z = p_.apply_right_inv(x);
    Matrix v_row(1, v_.rows());
    v_row.row(0) = v_.transpose();
    const Matrix vtpinv = p_.apply_right_inv(v_row);
    const Matrix correction = (z * u_) * (vtpinv / denominator_);
    z -= correction;
    return z;
}

Matrix SDIMMask::materialize() const {
    Matrix eye = Matrix::Identity(dim(), dim());
    return apply_left(eye);
}

namespace {

WeightedEdge transform_share_edge(const WeightedEdge& edge,
                                  double value,
                                  const ScaledPermutation& left,
                                  const ScaledPermutation& right) {
    const int protected_row = left.inverse_permutation()[static_cast<std::size_t>(edge.row)];
    const int protected_col = right.inverse_permutation()[static_cast<std::size_t>(edge.col)];
    const double protected_value =
        left.scale()[static_cast<std::size_t>(protected_row)] * value /
        right.scale()[static_cast<std::size_t>(protected_col)];
    return {protected_row, protected_col, protected_value};
}

}  // namespace

ProtectedGraphShares protect_graph_edges(const Graph& graph,
                                         double confusion_rate,
                                         const ScaledPermutation& p1,
                                         const ScaledPermutation& p2,
                                         const ScaledPermutation& p4,
                                         const ScaledPermutation& p5,
                                         RandomEngine& rng) {
    if (confusion_rate < 0.0) {
        throw std::runtime_error("confusion-rate must be non-negative");
    }
    const int n = graph.num_nodes();
    std::unordered_set<std::uint64_t> support;
    std::vector<WeightedEdge> augmented = graph.edges();
    support.reserve(augmented.size() * 2 + 1);
    for (const auto& edge : augmented) {
        support.insert(edge_key(edge.row, edge.col, n));
    }

    const std::size_t confusion_edges =
        static_cast<std::size_t>(std::floor(confusion_rate * static_cast<double>(graph.raw_directed_edges())));
    std::size_t inserted = 0;
    const std::size_t max_possible = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (support.size() + confusion_edges > max_possible) {
        throw std::runtime_error("confusion-rate asks for more edges than the augmented support can hold");
    }
    while (inserted < confusion_edges) {
        const int row = static_cast<int>(rng.uniform_index(static_cast<std::uint64_t>(n)));
        const int col = static_cast<int>(rng.uniform_index(static_cast<std::uint64_t>(n)));
        const auto key = edge_key(row, col, n);
        if (support.find(key) != support.end()) {
            continue;
        }
        support.insert(key);
        augmented.push_back({row, col, 0.0});
        inserted += 1;
    }

    ProtectedGraphShares shares;
    shares.confusion_edges = confusion_edges;
    shares.a1.reserve(augmented.size());
    shares.a2.reserve(augmented.size());
    for (const auto& edge : augmented) {
        const double eta = rng.uniform(-0.01, 0.01);
        shares.a1.push_back(transform_share_edge(edge, 0.5 * edge.value + eta, p1, p2));
        shares.a2.push_back(transform_share_edge(edge, 0.5 * edge.value - eta, p4, p5));
    }
    return shares;
}

}  // namespace teegnn
