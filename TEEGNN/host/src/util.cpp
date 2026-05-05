#include "util.hpp"
#include "blocked_csc.h"
#include "csc_graph.h"
#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

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

// in real system, this shoule be replaced by real key
std::array<uint8_t, TEEGNN_AES128_KEY_LEN> test_key() {
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key{};
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<uint8_t>(i * 17U + 3U);
    }
    return key;
}

}  // namespace

ScaledPermutation::ScaledPermutation(std::vector<uint32_t> permutation, std::vector<double> scale)
    : permutation_(std::move(permutation)), scale_(std::move(scale)) {
    if (permutation_.size() != scale_.size()) {
        throw std::runtime_error("scaled permutation size mismatch");
    }
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
    }
}

ScaledPermutation::ScaledPermutation(ScaledPermutation&& other)
    : permutation_(std::move(other.permutation_)),
      scale_(std::move(other.scale_)) {}

ScaledPermutation& ScaledPermutation::operator=(ScaledPermutation&& other) {
    permutation_ = std::move(other.permutation_);
    scale_ = std::move(other.scale_);
    return *this;
}

ScaledPermutation ScaledPermutation::random(int dim, RandomEngine& rng) {
    std::vector<uint32_t> permutation(static_cast<std::size_t>(dim));
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

MaskPhaseResult run_mask_phase(const Dataset& dataset, const Options& options) {
    RandomEngine rng_data(options.seed_data, 0, "teegnn-data-owner");
    RandomEngine rng_model(options.seed_model, 0, "teegnn-model-owner");                       
    
    MaskPhaseResult result;
    Secrets& secrets = result.secrets;
    MaskedData& masked_data = result.data;

    masked_data.num_nodes = dataset.graph.num_nodes();
    masked_data.feature_dim = static_cast<int>(dataset.features.cols());
    masked_data.hidden_dim = static_cast<int>(dataset.w1.cols());
    masked_data.class_dim = static_cast<int>(dataset.w2.cols());
    uint32_t block_size = masked_data.num_nodes / 64u *64;

    // data owner
    {
        SDIMMask s_right = SDIMMask::random(masked_data.feature_dim, rng_data);
        SDIMMask s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        masked_data.features = apply_SDIM(s_left, s_right, dataset.features);
        CSCGraph* g1 = new CSCGraph;
        CSCGraph* g2 = new CSCGraph;

        graph_to_csc_graph(dataset.graph, s_left.permutation(), g1);
        EncryptedBlockedCSC* enc1 = new EncryptedBlockedCSC;
        const auto key1 = test_key();
        blocked_csc_encrypt(
            g1, 
            0, 
            0, 
            0, 
            0, 
            block_size, 
            key1.data(), 
            key1.size(), 
            &enc1
        );
        masked_data.graphs.push_back(*enc1);

        s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        graph_to_csc_graph(dataset.graph, s_left.permutation(), g2);
        EncryptedBlockedCSC* enc2 = new EncryptedBlockedCSC;
        const auto key2 = test_key();
        blocked_csc_encrypt(
            g2, 
            0, 
            0, 
            1, 
            1, 
            block_size, 
            key2.data(), 
            key2.size(), 
            &enc2
        );
        masked_data.graphs.push_back(*enc2);
    }

    // model owner
    {
        masked_data.weights.push_back(dataset.w1);
        SDIMMask s_left = SDIMMask::random(masked_data.hidden_dim, rng_model);
        SDIMMask s_right = SDIMMask::random(masked_data.class_dim, rng_model);
        masked_data.weights.push_back(apply_SDIM(s_left, s_right, dataset.w2));
    }

    return result;
}

}  // namespace teegnn
