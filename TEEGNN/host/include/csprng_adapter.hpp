#pragma once

#include "types.hpp"

#include <cstdint>
#include <string>

extern "C" {
#include "csprng.h"
}

namespace teegnn {

class RandomEngine {
public:
    static constexpr int32_t kMatrixRandomMin = -(1 << 15);
    static constexpr int32_t kMatrixRandomMax = 1 << 15;

    RandomEngine(uint64_t seed, uint64_t stream_id, const std::string& label);
    ~RandomEngine();

    RandomEngine(const RandomEngine&) = delete;
    RandomEngine& operator=(const RandomEngine&) = delete;

    double uniform(double a, double b);
    int32_t uniform_int(int32_t min_inclusive, int32_t max_exclusive);
    uint32_t uniform_index(uint32_t n);
    int32_t nonzero_scale();

private:
    void check(int status, const char* what);

    csprng_master_t master_{};
    csprng_stream_t stream_{};
};

}  // namespace teegnn
