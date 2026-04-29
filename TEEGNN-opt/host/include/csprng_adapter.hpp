#pragma once

#include "types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
#include "csprng.h"
}

namespace teegnn {

class RandomEngine {
public:
    static constexpr std::int32_t kMatrixRandomMin = -(1 << 15);
    static constexpr std::int32_t kMatrixRandomMax = 1 << 15;

    RandomEngine(std::uint64_t seed, std::uint64_t stream_id, const std::string& label);
    ~RandomEngine();

    RandomEngine(const RandomEngine&) = delete;
    RandomEngine& operator=(const RandomEngine&) = delete;

    double uniform(double a, double b);
    std::int32_t uniform_int(std::int32_t min_inclusive, std::int32_t max_exclusive);
    std::uint32_t uniform_index(std::uint32_t n);
    double random_matrix_value();
    double nonzero_scale();
    Matrix random_matrix(int rows, int cols);

private:
    void check(int status, const char* what);

    csprng_master_t master_{};
    csprng_stream_t stream_{};
};

}  // namespace teegnn
