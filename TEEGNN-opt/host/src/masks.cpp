#include "masks.hpp"
#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <iostream>

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

ScaledPermutation::ScaledPermutation(ScaledPermutation&& other)
    : permutation_(std::move(other.permutation_)),
      inverse_permutation_(std::move(other.inverse_permutation_)),
      scale_(std::move(other.scale_)) {}

ScaledPermutation& ScaledPermutation::operator=(ScaledPermutation&& other) {
    permutation_ = std::move(other.permutation_);
    inverse_permutation_ = std::move(other.inverse_permutation_);
    scale_ = std::move(other.scale_);
    return *this;
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

Matrix apply_SPM(const ScaledPermutation& L, const ScaledPermutation& R, const Matrix& x) {
    int r = L.dim();
    int c = R.dim();
    Matrix out(x.rows(), x.cols());
    for (int j = 0; j < c; ++j) {
        int n_j = R.permutation()[j];
        for (int i = 0; i < r; ++i) {
            int n_i = L.permutation()[i];
            out(i, j) = x(n_i, n_j) / R.scale()[j] * L.scale()[i];
        }
    }
    return out;
}

Matrix apply_SPM_inv(const ScaledPermutation& L, const ScaledPermutation& R, const Matrix& x) {
    int r = L.dim();
    int c = R.dim();
    Matrix out(x.rows(), x.cols());
    for (int j = 0; j < c; ++j) {
        int n_j = R.permutation()[j];
        for (int i = 0; i < r; ++i) {
            int n_i = L.permutation()[i];
            out(n_i, n_j) = x(i, j) / L.scale()[i] * R.scale()[j];
        }
    }
    return out;
}

SDIMMask::SDIMMask(ScaledPermutation p, Vector h)
    : p_(std::move(p)), h_(std::move(h)) {
    if (h_.size() != p_.dim()) {
        throw std::runtime_error("SDIM vector dimension mismatch");
    }
    denominator_ = 1.0;
    for (int i = 0; i < p_.dim(); ++i) {
        denominator_ +=  h_(i) / p_.scale()[static_cast<std::size_t>(i)];
    }
    if (std::abs(denominator_) < 1e-8) {
        throw std::runtime_error("singular SDIM mask");
    }
}

SDIMMask::SDIMMask(SDIMMask&& other) {
    p_ = std::move(other.p_);
    h_ = std::move(other.h_);
    denominator_ = other.denominator_;
}

SDIMMask& SDIMMask::operator=(SDIMMask&& other) {
    p_ = std::move(other.p_);
    h_ = std::move(other.h_);
    denominator_ = other.denominator_;
    return *this;
}

SDIMMask SDIMMask::random(int dim, RandomEngine& rng) {
    for (int attempt = 0; attempt < 32; ++attempt) {
        ScaledPermutation p = ScaledPermutation::random(dim, rng);
        Vector h(dim);
        for (int i = 0; i < dim; ++i) {
            h(i) = rng.nonzero_scale();
        }
        try {
            return SDIMMask(std::move(p), std::move(h));
        } catch (const std::runtime_error&) {
        }
    }
    throw std::runtime_error("failed to generate non-singular SDIM mask");
}

Matrix apply_SDIM(const SDIMMask& L, const SDIMMask& R, const Matrix& x) {
    int r = L.dim();
    int c = R.dim();
    double sum = 0.0;
    Vector row_sum = Vector::Zero(r);
    Vector col_sum = Vector::Zero(c);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            row_sum(i) += x(i, R.perm(j)) / R.value(j) * R.h(j) ;
            col_sum(j) += x(i, j);
        }
    }
    for (int i = 0; i < r; ++i) {
        row_sum(i) /= R.denominator();
    }
    for (int j = 0; j < c; ++j) {
        sum += col_sum(R.perm(j)) / R.denominator() / R.value(j) * R.h(j);
    }

    Matrix out(x.rows(), x.cols());
    for (int j = 0; j < c; ++j) {
        int n_j = R.perm(j);
        for (int i = 0; i < r; ++i) {
            int n_i = L.perm(i);
            out(i, j) = x(n_i, n_j) / R.value(j) * L.value(i) + 
                                 col_sum(n_j) / R.value(j) * L.h(i) -
                                 row_sum(n_i) / R.value(j) * L.value(i) -
                                 sum / R.value(j) * L.h(i) ;
        }
    }
    // std::cout<< out.maxCoeff() <<' '<< out.minCoeff() <<'\n';
    return out;
}

Matrix apply_SDIM_inv(const SDIMMask& L, const SDIMMask& R, const Matrix& x) {
    int r = L.dim();
    int c = R.dim();
    double sum = 0.0;
    Vector row_sum = Vector::Zero(r);
    Vector col_sum = Vector::Zero(c);
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            row_sum(i) += x(i, j) / L.value(i) * R.h(j);
            col_sum(j) += x(i, j) / L.value(i) * R.value(j);
        }
        col_sum(j) /= L.denominator();
    }
    for (int i = 0; i < r; ++i) {
        sum += row_sum(i) / L.denominator();
    }
    // std::cout<< sum <<' '<< L.denominator() <<'\n';

    Matrix out(x.rows(), x.cols());
    for (int j = 0; j < c; ++j) {
        int n_j = R.perm(j);
        for (int i = 0; i < r; ++i) {
            int n_i = L.perm(i);
            out(n_i, n_j) = x(i, j) / L.value(i) * R.value(j) + 
                                     row_sum(i) -
                                     col_sum(j) / L.value(i) * L.h(i) -
                                     sum / L.value(i) * L.h(i);
        }
    }
    // std::cout<< out.maxCoeff() <<' '<< out.minCoeff() <<'\n';
    return out;
}

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
    shares.a1.assign(static_cast<std::size_t>(n), {});
    shares.a2.assign(static_cast<std::size_t>(n), {});
    for (const auto& edge : augmented) {
        const double eta = rng.random_matrix_value();
        const WeightedEdge a1_edge = transform_share_edge(edge, 0.5 * edge.value + eta, p1, p2);
        const WeightedEdge a2_edge = transform_share_edge(edge, 0.5 * edge.value - eta, p4, p5);
        shares.a1[static_cast<std::size_t>(a1_edge.row)].push_back({a1_edge.col, a1_edge.value});
        shares.a2[static_cast<std::size_t>(a2_edge.row)].push_back({a2_edge.col, a2_edge.value});
    }
    return shares;
}

MaskPhaseResult run_mask_phase(const Dataset& dataset, const Options& options) {
    RandomEngine rng_spm(options.seed, 0, "teegnn-spm");
    RandomEngine rng_lmm(options.seed, 1, "teegnn-lmm");
    RandomEngine rng_sdim(options.seed, 2, "teegnn-sdim");                       
    
    MaskPhaseResult result;
    MaskMatrices& matrices = result.matrices;
    MaskedData& masked_data = result.data;

    matrices.node_count = dataset.graph.num_nodes();
    matrices.feature_dim = static_cast<int>(dataset.features.cols());
    matrices.hidden_dim = static_cast<int>(dataset.w1.cols());
    const int class_dim = static_cast<int>(dataset.w2.cols());

    matrices.spm_n.emplace_back(ScaledPermutation::random(matrices.node_count, rng_spm));
    matrices.spm_n.emplace_back(ScaledPermutation::random(matrices.node_count, rng_spm));
    matrices.spm_n.emplace_back(ScaledPermutation::random(matrices.node_count, rng_spm));
    matrices.spm_n.emplace_back(ScaledPermutation::random(matrices.node_count, rng_spm));

    masked_data.graph_shares =
        protect_graph_edges(dataset.graph, options.confusion_rate, 
                            matrices.spm_n[0], matrices.spm_n[1],
                            matrices.spm_n[2], matrices.spm_n[3], rng_spm);

    matrices.precompute_Ahat_u.emplace_back(sparse_dense_mul(dataset.graph, 
                rng_lmm.random_matrix(matrices.node_count, options.mask_rank)));  
    matrices.precompute_Ahat_u.emplace_back(sparse_dense_mul(dataset.graph, 
                rng_lmm.random_matrix(matrices.node_count, options.mask_rank)));  
    
    {
        SDIMMask s_right = SDIMMask::random(matrices.feature_dim, rng_sdim);
        SDIMMask s_left = SDIMMask::random(matrices.node_count, rng_sdim);
        masked_data.features = apply_SDIM(s_left, s_right, dataset.features);
        matrices.sdim_masks.push_back(std::move(s_left));
        matrices.sdim_masks.push_back(std::move(s_right));
    }
    masked_data.weights.push_back(dataset.w1);
    {
        SDIMMask s_left = SDIMMask::random(matrices.hidden_dim, rng_sdim);
        SDIMMask s_right = SDIMMask::random(class_dim, rng_sdim);
        masked_data.weights.push_back(apply_SDIM(s_left, s_right, dataset.w2));
        matrices.sdim_masks.push_back(std::move(s_left));
        matrices.sdim_masks.push_back(std::move(s_right));
    }

    return result;
}

}  // namespace teegnn
