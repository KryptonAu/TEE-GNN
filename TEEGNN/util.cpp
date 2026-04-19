#include "util.h"

#include <cstdlib>

typedef struct PRNG {
    uint32_t state;
} PRNG;

void prng_seed(PRNG *rng, uint32_t seed) {
    assert(rng != 0);
    rng->state = seed;
}

PRNG prng_make(uint32_t seed) {
    PRNG rng;
    prng_seed(&rng, seed);
    return rng;
}

uint32_t prng_next_u32(PRNG *rng) {
    assert(rng != 0);
    // 经典 LCG 参数，基于 32 位无符号整数运算，平台间结果稳定。
    rng->state = rng->state * 1664525u + 1013904223u;
    return rng->state;
}

int prng_next_int(PRNG *rng, int min_value, int max_value) {
    assert(rng != 0);
    assert(min_value <= max_value);

    uint64_t span = (uint64_t)((int64_t)max_value - (int64_t)min_value) + 1u;
    if (span >= 4294967296ull) {
        return (int)prng_next_u32(rng);
    }

    // 拒绝采样，避免简单取模带来的分布偏差。
    uint32_t span_u32 = (uint32_t)span;
    uint32_t limit = UINT32_MAX - (UINT32_MAX % span_u32);
    uint32_t value = prng_next_u32(rng);
    while (value >= limit) {
        value = prng_next_u32(rng);
    }

    return min_value + (int)(value % span_u32);
}

double prng_next_double(PRNG *rng) {
    assert(rng != 0);
    return (double)prng_next_u32(rng) / 4294967296.0;
}

uint32_t generate_rrng_seed(PRNG *rng) {
    assert(rng != 0);

    uint32_t seed = prng_next_u32(rng);
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    seed ^= seed >> 16;

    if (seed == 0u) {
        seed = 0x6d2b79f5u;
    }

    return seed;
}

void random_permutation(PRNG *rng, int n, int *perm) {
    assert(rng != 0);
    assert(n >= 0);
    assert(perm != 0);

    for (int i = 0; i < n; ++i) {
        perm[i] = i;
    }

    for (int i = n - 1; i > 0; --i) {
        int j = prng_next_int(rng, 0, i);
        std::swap(perm[i], perm[j]);
    }
}

void random_vector(PRNG *rng, int n, int min_value, int max_value, int *vec) {
    assert(rng != 0);
    assert(n >= 0);
    assert(vec != 0);
    assert(min_value <= max_value);
    assert(!(min_value == 0 && max_value == 0));

    for (int i = 0; i < n; ++i) {
        int value = 0;
        do {
            value = prng_next_int(rng, min_value, max_value);
        } while (value == 0);
        vec[i] = value;
    }
}

// Random Permutation Matrix
typedef struct RPM {
    int n;
    int *perm;
    int *value;
} RPM;
void generate_rpm(PRNG *rng, int n, RPM *rpm) {
    assert(rng != 0);
    assert(rpm != 0);
    assert(n >= 0);

    rpm->n = n;
    rpm->perm = static_cast<int *>(std::malloc(static_cast<size_t>(n) * sizeof(int)));
    rpm->value = static_cast<int *>(std::malloc(static_cast<size_t>(n) * sizeof(int)));

    assert(rpm->perm != 0 || n == 0);
    assert(rpm->value != 0 || n == 0);

    random_permutation(rng, n, rpm->perm);
    random_vector(rng, n, -1000, 1000, rpm->value);
}

// Subtly Designed Invertible Matrix
typedef struct SDIM {
    int n;
    int *perm;
    int *value;
    int *h; // 加性扰动向量，长度为 n
} SDIM;
void generate_sdim(PRNG *rng, int n, SDIM *sdim) {
    assert(rng != 0);
    assert(sdim != 0);
    assert(n >= 0);

    sdim->n = n;
    sdim->perm = static_cast<int *>(std::malloc(static_cast<size_t>(n) * sizeof(int)));
    sdim->value = static_cast<int *>(std::malloc(static_cast<size_t>(n) * sizeof(int)));
    sdim->h = static_cast<int *>(std::malloc(static_cast<size_t>(n) * sizeof(int)));

    assert(sdim->perm != 0 || n == 0);
    assert(sdim->value != 0 || n == 0);
    assert(sdim->h != 0 || n == 0);

    random_permutation(rng, n, sdim->perm);
    random_vector(rng, n, -1000, 1000, sdim->value);
    random_vector(rng, n, -1000, 1000, sdim->h);
}

// Low-rank Mask Matrix
typedef struct LMM {
    int n;
    int rank;
    int *u; // 左奇异向量，长度为 n*rank
    int *v; // 右奇异向量，长度为 n*rank
} LMM;
void generate_lmm(PRNG *rng, int n, int rank, LMM *lmm) {
    assert(rng != 0);
    assert(lmm != 0);
    assert(n >= 0);
    assert(rank >= 0);

    lmm->n = n;
    lmm->rank = rank;

    size_t total = static_cast<size_t>(n) * static_cast<size_t>(rank);
    lmm->u = static_cast<int *>(std::malloc(total * sizeof(int)));
    lmm->v = static_cast<int *>(std::malloc(total * sizeof(int)));

    assert(lmm->u != 0 || total == 0);
    assert(lmm->v != 0 || total == 0);

    random_vector(rng, static_cast<int>(total), -1000, 1000, lmm->u);
    random_vector(rng, static_cast<int>(total), -1000, 1000, lmm->v);
}
