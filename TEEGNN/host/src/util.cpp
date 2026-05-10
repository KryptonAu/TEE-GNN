#include "util.hpp"
#include "blocked_csc.h"
#include "csc_graph.h"
#include "graph.hpp"
#include "teegnn_error.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace teegnn {
namespace {

// in real system, this shoule be replaced by real key
std::array<uint8_t, TEEGNN_AES128_KEY_LEN> test_key() {
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key{};
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<uint8_t>(i * 17U + 3U);
    }
    return key;
}

}  // namespace

SDIMMask::SDIMMask(std::vector<uint32_t> p, std::vector<int32_t> v, std::vector<int32_t> h) 
    : p_(std::move(p)), v_(std::move(v)), h_(std::move(h)) {
    if (h_.size() != p_.size()) {
        throw std::runtime_error("SDIM vector dimension mismatch");
    }
    denominator_ = 1.0;
    for (size_t i = 0; i < h_.size(); ++i) {
        denominator_ +=  static_cast<double>(h_[i]) / v_[i];
    }
    if (std::abs(denominator_) < 1e-8) {
        throw std::runtime_error("singular SDIM mask");
    }
}

SDIMMask::SDIMMask(SDIMMask&& other) {
    p_ = std::move(other.p_);
    v_ = std::move(other.v_);
    h_ = std::move(other.h_);
    denominator_ = other.denominator_;
}

SDIMMask& SDIMMask::operator=(SDIMMask&& other) {
    p_ = std::move(other.p_);
    v_ = std::move(other.v_);
    h_ = std::move(other.h_);
    denominator_ = other.denominator_;
    return *this;
}

SDIMMask SDIMMask::random(size_t dim, RandomEngine& rng) {
    for (int attempt = 0; attempt < 32; ++attempt) {
        std::vector<uint32_t> permutation(dim);
        std::vector<int32_t> value(dim);
        std::vector<int32_t> h(dim);
        for (size_t i = 0; i < dim; ++i) {
            permutation[i] = i;
            value[i] = rng.nonzero_scale();
        }
        for (size_t i = dim - 1; i > 0; --i) {
            const uint32_t j = rng.uniform_index(static_cast<uint32_t>(i + 1));
            std::swap(permutation[i], permutation[j]);
        }
        for (size_t i = 0; i < dim; ++i) {
            h[i] = rng.nonzero_scale();
        }
        try {
            return SDIMMask(std::move(permutation), std::move(value),std::move(h));
        } catch (const std::runtime_error&) {
        }
    }
    throw std::runtime_error("failed to generate non-singular SDIM mask");
}

Matrix apply_SDIM(const SDIMMask& L, const SDIMMask& R, const Matrix& x) {
    size_t r = L.dim();
    size_t c = R.dim();
    double sum = 0.0;
    Vector row_sum = Vector::Zero(r);
    Vector col_sum = Vector::Zero(c);
    for (size_t j = 0; j < c; ++j) {
        for (size_t i = 0; i < r; ++i) {
            row_sum(i) += x(i, R.perm(j)) / R.value(j) * R.h(j) ;
            col_sum(j) += x(i, j);
        }
    }
    for (size_t i = 0; i < r; ++i) {
        row_sum(i) /= R.denominator();
    }
    for (size_t j = 0; j < c; ++j) {
        sum += col_sum(R.perm(j)) / R.denominator() / R.value(j) * R.h(j);
    }

    Matrix out(x.rows(), x.cols());
    for (size_t j = 0; j < c; ++j) {
        size_t n_j = R.perm(j);
        for (size_t i = 0; i < r; ++i) {
            size_t n_i = L.perm(i);
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
    size_t r = L.dim();
    size_t c = R.dim();
    double sum = 0.0;
    Vector row_sum = Vector::Zero(r);
    Vector col_sum = Vector::Zero(c);
    for (size_t j = 0; j < c; ++j) {
        for (size_t i = 0; i < r; ++i) {
            row_sum(i) += x(i, j) / L.value(i) * R.h(j);
            col_sum(j) += x(i, j) / L.value(i) * R.value(j);
        }
        col_sum(j) /= L.denominator();
    }
    for (size_t i = 0; i < r; ++i) {
        sum += row_sum(i) / L.denominator();
    }
    // std::cout<< sum <<' '<< L.denominator() <<'\n';

    Matrix out(x.rows(), x.cols());
    for (size_t j = 0; j < c; ++j) {
        size_t n_j = R.perm(j);
        for (size_t i = 0; i < r; ++i) {
            size_t n_i = L.perm(i);
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
    teegnn_status_t st;

    secrets.seed_data = options.seed_data;
    secrets.seed_model = options.seed_model;
    masked_data.num_nodes = dataset.graph.num_nodes();
    masked_data.feature_dim = static_cast<int>(dataset.features.cols());
    masked_data.hidden_dim = static_cast<int>(dataset.w1.cols());
    masked_data.class_dim = static_cast<int>(dataset.w2.cols());
    uint32_t block_size = (masked_data.num_nodes + 63u) / 64u *64;

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
        st = blocked_csc_encrypt(
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
        if (st != TEEGNN_OK) {
            throw std::runtime_error("Failed to encrypt csc");
        }
        secrets.key1 = key1;
        masked_data.graphs.push_back(std::unique_ptr<EncryptedBlockedCSC>(enc1));

        s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        graph_to_csc_graph(dataset.graph, s_left.permutation(), g2);
        EncryptedBlockedCSC* enc2 = new EncryptedBlockedCSC;
        const auto key2 = test_key();
        st = blocked_csc_encrypt(
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
        if (st != TEEGNN_OK) {
            throw std::runtime_error("Failed to encrypt csc");
        }
        secrets.key2 = key2;
        masked_data.graphs.push_back(std::unique_ptr<EncryptedBlockedCSC>(enc2));

        free(g1);
        free(g2);
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
