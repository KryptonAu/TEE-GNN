#pragma once

#include "csprng_adapter.hpp"
#include "graph.hpp"
#include "types.hpp"
#include "dataset_loader.hpp"

#include <utility>
#include <vector>

namespace teegnn {

class ScaledPermutation {
public:
    ScaledPermutation() = default;
    ScaledPermutation(std::vector<int> permutation, std::vector<double> scale);
    ScaledPermutation(ScaledPermutation&& other);
    ScaledPermutation& operator=(ScaledPermutation&& other);

    static ScaledPermutation random(int dim, RandomEngine& rng);

    int dim() const { return static_cast<int>(permutation_.size()); }
    const std::vector<int>& permutation() const { return permutation_; }
    const std::vector<int>& inverse_permutation() const { return inverse_permutation_; }
    const std::vector<double>& scale() const { return scale_; }

    Matrix apply_left(const Matrix& x) const;
    Matrix apply_left_inv(const Matrix& x) const;
    Matrix apply_right(const Matrix& x) const;
    Matrix apply_right_inv(const Matrix& x) const;

private:
    std::vector<int> permutation_;
    std::vector<int> inverse_permutation_;
    std::vector<double> scale_;
};

Matrix apply_SPM(const ScaledPermutation& L, const ScaledPermutation& R, const Matrix& x);

Matrix apply_SPM_inv(const ScaledPermutation& L, const ScaledPermutation& R, const Matrix& x);

struct LowRankMask {
    Matrix u;
    Matrix v;

    Matrix materialize() const;
    Matrix ahat_times_mask(const Graph& graph) const;
};

LowRankMask make_low_rank_mask(int rows, int cols, int rank, RandomEngine& rng);

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
    int perm(int index) const { return p_.permutation()[static_cast<std::size_t>(index)]; }
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
    double confusion_rate = 0.0;
    int mask_rank = 2;
    std::uint64_t seed = 1;
};

struct ProtectedGraphShares {
    std::vector<std::vector<std::pair<int, double>>> a1;
    std::vector<std::vector<std::pair<int, double>>> a2;
    std::size_t confusion_edges = 0;
};

ProtectedGraphShares protect_graph_edges(const Graph& graph,
                                         double confusion_rate,
                                         const ScaledPermutation& p1,
                                         const ScaledPermutation& p2,
                                         const ScaledPermutation& p4,
                                         const ScaledPermutation& p5,
                                         RandomEngine& rng);

struct MaskedFeature {
    teegnn::Matrix share1;
    teegnn::Matrix share2;
};

struct MaskMatrices {
    int node_count = 0;
    int feature_dim = 0;
    int hidden_dim = 0;

    teegnn::ScaledPermutation p1;
    teegnn::ScaledPermutation p2;
    teegnn::ScaledPermutation p3;
    teegnn::ScaledPermutation p4;
    teegnn::ScaledPermutation p5;
    teegnn::ScaledPermutation p6;

    std::vector<teegnn::LowRankMask> lr_masks;
    std::vector<teegnn::Matrix> precompute_Ahat_u;
    std::vector<teegnn::SDIMMask> sdim_masks;
};

struct MaskedData {
    teegnn::ProtectedGraphShares graph_shares;
    MaskedFeature features;
    std::vector<teegnn::Matrix> weights;
};

struct MaskPhaseResult {
    MaskMatrices matrices;
    MaskedData data;
};

MaskedFeature feature_mask(const teegnn::Matrix& x,
                                             const teegnn::LowRankMask& low_rank_mask,
                                             const teegnn::ScaledPermutation& left1,
                                             const teegnn::ScaledPermutation& right1,
                                             const teegnn::ScaledPermutation& left2,
                                             const teegnn::ScaledPermutation& right2);

teegnn::MaskPhaseResult run_mask_phase(const teegnn::Dataset& dataset,
                                       const Options& options);

}  // namespace teegnn
