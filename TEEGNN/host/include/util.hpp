#pragma once

#include "csprng_adapter.hpp"
#include "types.hpp"
#include "dataset_loader.hpp"
#include "blocked_csc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace teegnn {

class ScaledPermutation {
public:
    ScaledPermutation() = default;
    ScaledPermutation(std::vector<uint32_t> permutation, std::vector<double> scale);
    ScaledPermutation(ScaledPermutation&& other);
    ScaledPermutation& operator=(ScaledPermutation&& other);

    static ScaledPermutation random(int dim, RandomEngine& rng);

    int dim() const { return static_cast<int>(permutation_.size()); }
    const std::vector<uint32_t>& permutation() const { return permutation_; }
    const std::vector<double>& scale() const { return scale_; }

private:
    std::vector<uint32_t> permutation_;
    std::vector<double> scale_;
};

class SDIMMask {
public:
    SDIMMask() = default;
    SDIMMask(ScaledPermutation p, Vector h);
    SDIMMask(const SDIMMask& other) = delete;
    SDIMMask(SDIMMask&& other);
    SDIMMask& operator=(SDIMMask&& other);

    static SDIMMask random(int dim, RandomEngine& rng);

    int dim() const { return p_.dim(); }
    double value(int index) const { return p_.scale()[static_cast<std::size_t>(index)]; }
    double h(int index) const { return h_(static_cast<std::size_t>(index)); }
    uint32_t perm(int index) const { return p_.permutation()[static_cast<std::size_t>(index)]; }
    const std::vector<uint32_t>& permutation() const { return p_.permutation(); }
    double denominator() const { return denominator_; }

private:
    ScaledPermutation p_;
    Vector h_;
    double denominator_ = 1.0;
};

Matrix apply_SDIM(const SDIMMask& L, const SDIMMask& R, const Matrix& x);

Matrix apply_SDIM_inv(const SDIMMask& L, const SDIMMask& R, const Matrix& x);

struct Options {
    std::string dataset_dir;
    std::uint64_t seed_data = 1234;
    std::uint64_t seed_model = 4321;
};

struct Secrets {
    std::uint64_t seed_data = 1;
    std::uint64_t seed_model = 1;
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key1;
    std::array<uint8_t, TEEGNN_AES128_KEY_LEN> key2;
};

struct MaskedData {
    size_t num_nodes;
    size_t feature_dim;
    size_t hidden_dim;
    size_t class_dim;
    std::vector<std::unique_ptr<EncryptedBlockedCSC>> graphs;
    Matrix features;
    std::vector<Matrix> weights;
};

struct MaskPhaseResult {
    Secrets secrets;
    MaskedData data;
};

MaskPhaseResult run_mask_phase(const Dataset& dataset, const Options& options);

}  // namespace teegnn
