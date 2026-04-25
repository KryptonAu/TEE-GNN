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
    RandomEngine(std::uint64_t seed, std::uint64_t stream_id, const std::string& label);
    ~RandomEngine();

    RandomEngine(const RandomEngine&) = delete;
    RandomEngine& operator=(const RandomEngine&) = delete;

    double uniform(double a, double b);
    std::uint64_t uniform_index(std::uint64_t n);
    double nonzero_scale();
    Matrix normal_like(int rows, int cols, double amplitude);

private:
    void check(int status, const char* what);

    csprng_master_t master_{};
    csprng_stream_t stream_{};
};

}  // namespace teegnn

