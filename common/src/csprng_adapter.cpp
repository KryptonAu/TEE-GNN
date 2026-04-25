#include "csprng_adapter.hpp"

#include <cstring>

namespace teegnn {

RandomEngine::RandomEngine(std::uint64_t seed, std::uint64_t stream_id, const std::string& label) {
    check(csprng_master_init(&master_, &seed, sizeof(seed)), "csprng_master_init");
    check(csprng_stream_init(&stream_, &master_, stream_id, label.data(), label.size()), "csprng_stream_init");
}

RandomEngine::~RandomEngine() {
    csprng_stream_wipe(&stream_);
    csprng_master_wipe(&master_);
}

double RandomEngine::uniform(double a, double b) {
    double out = 0.0;
    check(csprng_double_range(&stream_, a, b, &out), "csprng_double_range");
    return out;
}

std::uint64_t RandomEngine::uniform_index(std::uint64_t n) {
    if (n == 0) {
        throw std::runtime_error("uniform_index called with zero range");
    }
    std::uint64_t out = 0;
    check(csprng_u64_range(&stream_, 0, n, &out), "csprng_u64_range");
    return out;
}

double RandomEngine::nonzero_scale() {
    double value = 0.0;
    do {
        value = uniform(-2.0, 2.0);
    } while (value > -0.05 && value < 0.05);
    return value;
}

Matrix RandomEngine::normal_like(int rows, int cols, double amplitude) {
    Matrix out(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out(i, j) = uniform(-amplitude, amplitude);
        }
    }
    return out;
}

void RandomEngine::check(int status, const char* what) {
    if (status != CSPRNG_OK) {
        throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(status));
    }
}

}  // namespace teegnn

