#pragma once

#include "csprng_adapter.hpp"
#include "graph.hpp"
#include "types.hpp"

#include <vector>

namespace teegnn {

class ScaledPermutation {
public:
    ScaledPermutation() = default;
    ScaledPermutation(std::vector<int> permutation, std::vector<double> scale);

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

struct LowRankMask {
    Matrix u;
    Matrix v;

    Matrix materialize() const;
    Matrix ahat_times_mask(const Graph& graph) const;
};

LowRankMask make_low_rank_mask(int rows, int cols, int rank, RandomEngine& rng, double amplitude = 0.02);

class SDIMMask {
public:
    SDIMMask() = default;
    SDIMMask(ScaledPermutation p, Vector u, Vector v);

    static SDIMMask random(int dim, RandomEngine& rng);

    int dim() const { return p_.dim(); }
    Matrix apply_left(const Matrix& x) const;
    Matrix apply_left_inv(const Matrix& x) const;
    Matrix apply_right(const Matrix& x) const;
    Matrix apply_right_inv(const Matrix& x) const;
    Matrix materialize() const;

private:
    ScaledPermutation p_;
    Vector u_;
    Vector v_;
    double denominator_ = 1.0;
};

struct ProtectedGraphShares {
    std::vector<WeightedEdge> a1;
    std::vector<WeightedEdge> a2;
    std::size_t confusion_edges = 0;
};

ProtectedGraphShares protect_graph_edges(const Graph& graph,
                                         double confusion_rate,
                                         const ScaledPermutation& p1,
                                         const ScaledPermutation& p2,
                                         const ScaledPermutation& p4,
                                         const ScaledPermutation& p5,
                                         RandomEngine& rng);

}  // namespace teegnn
