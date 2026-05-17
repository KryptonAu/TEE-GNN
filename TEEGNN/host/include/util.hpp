#pragma once

#include "csprng_adapter.hpp"
#include "types.hpp"
#include "dataset_loader.hpp"
#include "blocked_edge_list.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace teegnn {

#define TA_DATA_SIZE (1 << 22) // 4 * 1024 * 1024 Byte

class SDIMMask {
public:
    SDIMMask() = default;
    SDIMMask(std::vector<uint32_t> p, std::vector<int32_t> v, std::vector<int32_t> h);
    SDIMMask(const SDIMMask& other) = delete;
    SDIMMask(SDIMMask&& other);
    SDIMMask& operator=(SDIMMask&& other);

    static SDIMMask random(size_t dim, RandomEngine& rng);

    size_t dim() const { return p_.size(); }
    int32_t value(int index) const { return v_[index]; }
    int32_t h(int index) const { return h_[index]; }
    uint32_t perm(int index) const { return p_[index]; }
    const std::vector<uint32_t>& permutation() const { return p_; }
    double denominator() const { return denominator_; }

private:
    std::vector<uint32_t> p_;
    std::vector<int32_t> v_;
    std::vector<int32_t> h_;
    double denominator_ = 1.0;
};

Matrix apply_SDIM(const SDIMMask& L, const SDIMMask& R, const Matrix& x);

Matrix apply_SDIM_inv(const SDIMMask& L, const SDIMMask& R, const Matrix& x);

struct Options {
    std::string dataset_dir;
    uint64_t seed_data = 1234;
    uint64_t seed_model = 4321;
};

struct Secrets {
    uint64_t seed_data = 1;
    uint64_t seed_model = 1;
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key1;
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key2;
    uint32_t row_block_size = 1;
    uint32_t edge_block_size = 1;
};

struct MaskedData {
    size_t num_nodes;
    size_t feature_dim;
    size_t hidden_dim;
    size_t class_dim;
    std::vector<std::unique_ptr<EncryptedBlockedEdgeList>> graphs;
    Matrix features;
    std::vector<Matrix> weights;
};

struct MaskPhaseResult {
    Secrets secrets;
    MaskedData data;
};

MaskPhaseResult run_mask_phase(const Dataset& dataset, const Options& options);

}  // namespace teegnn
