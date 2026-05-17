#include "util.hpp"
#include "blocked_edge_list.h"
#include "crypto.h"
#include "edge_list.h"
#include "graph.hpp"
#include "teegnn_error.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <utility>
#include <iostream>

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
    uint32_t row_block_size = block_size;

    // calculate the size of edge_list block and row block
    {
        uint32_t ta_data_size = TA_DATA_SIZE;

        // sdim
        ta_data_size -= (masked_data.num_nodes + masked_data.hidden_dim) * sizeof(int32_t) * 3;
        // col_sum, col_sum_out, temp_row
        ta_data_size -= masked_data.hidden_dim * 3 * sizeof(double);
        // keys
        ta_data_size -= TEEGNN_AES128_KEY_LEN * 3;

        ta_data_size -= 1 << 16; // 64KB for other data structure

        double a = std::sqrt((4 + 4 + 8 + 1) * dataset.graph.num_edges());
        double b = std::sqrt(masked_data.num_nodes * masked_data.hidden_dim * sizeof(double));

        block_size = static_cast<uint32_t>(ta_data_size / (a + b) * a) / 17;
        row_block_size = static_cast<uint32_t>(ta_data_size / (a + b) * b) / 8 / masked_data.hidden_dim;
        block_size = std::min((size_t)block_size, dataset.graph.num_edges());
        row_block_size = std::min((size_t)row_block_size, masked_data.num_nodes);

        size_t num_blocks = (dataset.graph.num_edges() + block_size - 1) / block_size;
        size_t num_row_blocks = (masked_data.num_nodes + row_block_size - 1) / row_block_size;
        block_size = (dataset.graph.num_edges() + num_blocks - 1) / num_blocks;
        row_block_size = (masked_data.num_nodes + num_row_blocks - 1) / num_row_blocks;
    }

    std::cout << block_size << ' ' << row_block_size << '\n';

    secrets.row_block_size = row_block_size;
    secrets.edge_block_size = block_size;

    // data owner
    {
        std::vector<SDIMMask> sdims;
        SDIMMask s_right = SDIMMask::random(masked_data.feature_dim, rng_data);
        SDIMMask s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        masked_data.features = apply_SDIM(s_left, s_right, dataset.features);
        EdgeList* g1 = new EdgeList;
        EdgeList* g2 = new EdgeList;

        sdims.push_back(std::move(s_left));
        sdims.push_back(std::move(s_right));

        // layer 1
        s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        sdims.push_back(std::move(s_left));

        // layer 2
        s_left = SDIMMask::random(masked_data.num_nodes, rng_data);
        sdims.push_back(std::move(s_left));

        graph_to_edge_list(
            dataset.graph, 
            sdims[0].permutation(), 
            sdims[2].permutation(), 
            row_block_size, 
            g1
        );
        EncryptedBlockedEdgeList* enc1 = nullptr;
        const auto key1 = test_key();
        st = blocked_edge_list_encrypt(
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
            throw std::runtime_error("Failed to encrypt edge_list");
        }
        secrets.key1 = key1;
        masked_data.graphs.push_back(std::unique_ptr<EncryptedBlockedEdgeList>(enc1));

        
        graph_to_edge_list(
            dataset.graph, 
            sdims[2].permutation(), 
            sdims[3].permutation(), 
            row_block_size, 
            g2
        );
        EncryptedBlockedEdgeList* enc2 = nullptr;
        const auto key2 = test_key();
        st = blocked_edge_list_encrypt(
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
            throw std::runtime_error("Failed to encrypt edge_list");
        }
        secrets.key2 = key2;
        masked_data.graphs.push_back(std::unique_ptr<EncryptedBlockedEdgeList>(enc2));

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

    std::cout << "Mask phase completed" << std::endl;

    return result;
}

}  // namespace teegnn
