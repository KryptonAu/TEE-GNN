#include "csprng_adapter.hpp"

#include <cmath>
#include <cstring>
#include <limits>

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

std::int32_t RandomEngine::uniform_int(std::int32_t min_inclusive, std::int32_t max_exclusive) {
    if (min_inclusive >= max_exclusive) {
        throw std::runtime_error("uniform_int called with an empty range");
    }
    std::uint32_t span = 0;
    if (min_inclusive < 0 && max_exclusive > 0) {
        const std::uint64_t negative_count =
            static_cast<std::uint64_t>(-(min_inclusive + 1)) + 1U;
        span = negative_count + static_cast<std::uint64_t>(max_exclusive);
        if (span == 0) {
            throw std::runtime_error("uniform_int range is too large");
        }
    } else {
        span = static_cast<std::uint32_t>(max_exclusive - min_inclusive);
    }

    std::uint32_t offset = 0;
    check(csprng_u32_range(&stream_, 0, span, &offset), "csprng_u32_range");
    if (min_inclusive < 0 && max_exclusive > 0) {
        const std::uint32_t negative_count =
            static_cast<std::uint32_t>(-(min_inclusive + 1)) + 1U;
        if (offset < negative_count) {
            return min_inclusive + static_cast<std::int32_t>(offset);
        }
        return static_cast<std::int32_t>(offset - negative_count);
    }
    return min_inclusive + static_cast<std::int32_t>(offset);
}

std::uint32_t RandomEngine::uniform_index(std::uint32_t n) {
    if (n == 0) {
        throw std::runtime_error("uniform_index called with zero range");
    }
    std::uint32_t out = 0;
    check(csprng_u32_range(&stream_, 0, n, &out), "csprng_u32_range");
    return out;
}

double RandomEngine::random_matrix_value() {
    return static_cast<double>(uniform_int(kMatrixRandomMin, kMatrixRandomMax));
}

double RandomEngine::nonzero_scale() {
    std::int32_t value = 0;
    do {
        value = uniform_int(kMatrixRandomMin, kMatrixRandomMax);
    } while (value == 0);
    return static_cast<double>(value);
}

Matrix RandomEngine::random_matrix(int rows, int cols) {
    Matrix out(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            out(i, j) = static_cast<double>(uniform_int(-256, 255));
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
