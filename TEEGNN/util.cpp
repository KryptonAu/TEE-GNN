#include "util.h"

#include <cstdlib>

const int MIN_RAND_VALUE = -(1 << 16);
const int MAX_RAND_VALUE = (1 << 16) - 1;

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
    random_vector(rng, n, MIN_RAND_VALUE, MAX_RAND_VALUE, rpm->value);
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
    random_vector(rng, n, MIN_RAND_VALUE, MAX_RAND_VALUE, sdim->value);
    random_vector(rng, n, MIN_RAND_VALUE, MAX_RAND_VALUE, sdim->h);
}

// Low-rank Mask Matrix
typedef struct LMM {
    int n;
    int m;
    int rank;
    int *u; // 左奇异向量，长度为 n*rank
    int *v; // 右奇异向量，长度为 m*rank
} LMM;
void generate_lmm(PRNG *rng, int n, int m, int rank, LMM *lmm) {
    assert(rng != 0);
    assert(lmm != 0);
    assert(n >= 0);
    assert(m >= 0);
    assert(rank >= 0);

    lmm->n = n;
    lmm->m = m;
    lmm->rank = rank;

    size_t total_n = static_cast<size_t>(n) * static_cast<size_t>(rank);
    size_t total_m = static_cast<size_t>(m) * static_cast<size_t>(rank);
    lmm->u = static_cast<int *>(std::malloc(total_n * sizeof(int)));
    lmm->v = static_cast<int *>(std::malloc(total_m * sizeof(int)));

    assert(lmm->u != 0 || total_n == 0);
    assert(lmm->v != 0 || total_m == 0);

    random_vector(rng, static_cast<int>(total_n), MIN_RAND_VALUE, MAX_RAND_VALUE, lmm->u);
    random_vector(rng, static_cast<int>(total_m), MIN_RAND_VALUE, MAX_RAND_VALUE, lmm->v);
}

void free_rpm(RPM *rpm) {
    if (rpm != 0) {
        std::free(rpm->perm);
        std::free(rpm->value);
    }
}
void free_sdim(SDIM *sdim) {
    if (sdim != 0) {
        std::free(sdim->perm);
        std::free(sdim->value);
        std::free(sdim->h);
    }
}
void free_lmm(LMM *lmm) {
    if (lmm != 0) {
        std::free(lmm->u);
        std::free(lmm->v);
    }
}


// matrices are in column-major order.
void weight_mask(double *in, double *out, SDIM *L, SDIM *R) {
    int r = L->n;
    int c = R->n;
    double factor = 1.0;
    double sum = 0.0;
    double* row_sum = static_cast<double *>(std::malloc(static_cast<size_t>(r) * sizeof(double)));
    double* col_sum = static_cast<double *>(std::malloc(static_cast<size_t>(c) * sizeof(double)));
    for (int j = 0; j < c; ++j) {
        factor += static_cast<double>(R->h[j]) / R->value[j];
        for (int i = 0; i < r; ++i) {
            row_sum[i] += in[i + R->perm[j] * r] * R->h[j] / R->value[j];
            col_sum[j] += in[i + j * r];
        }
    }
    for (int j = 0; j < c; ++j) {
        sum += col_sum[R->perm[j]] * R->h[j] / R->value[j];
    }
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            out[i + j * r] = in[L->perm[i] + R->perm[j] * r] * L->value[i] / R->value[j] + 
                             col_sum[R->perm[j]] * L->h[i] / R->value[j] -
                             row_sum[L->perm[i]] * L->value[i] / R->value[j] / factor -
                             sum * L->h[i] / R->value[j] / factor;
        }
    }
    free(row_sum);
    free(col_sum);
}

// X = L * (X + M) * R^-1
void feature_mask(double *in, double *out, RPM *L, RPM *R, LMM *M) {
    int r = L->n;
    int c = R->n;
    for (int j = 0; j < c; ++j) {
        int new_j = R->perm[j];
        for (int i = 0; i < r; ++i) {
            int new_i = L->perm[i];
            int mask_value = 0;
            for (int k = 0; k < M->rank; ++k) {
                mask_value += M->u[i + k * r] * M->v[j + k * c];
            }
            out[i + j * r] = (in[new_i + new_j * r] + mask_value) * L->value[new_i] / R->value[new_j];
        }
    }
}

void graph_split(PRNG *rng, const Weighted_Graph &g, Weighted_Graph &g1, Weighted_Graph &g2, 
                 const std::vector<RPM*> &rpms, double r) {
    int n = g.num_vertices();
    static std::set<std::pair<int, int>> edge_set;
    int* values[4] = {rpms[0]->value, rpms[1]->value, rpms[2]->value, rpms[3]->value};

    std::vector<std::vector<int>> inv_perms;
    for (int i = 0; i < 4; ++i) {
        std::vector<int> inv_perm(n);
        for (int j = 0; j < n; ++j) {
            inv_perm[rpms[i]->perm[j]] = j;
        }
        inv_perms.emplace_back(inv_perm);
    }
    
    int random_value;
    for (int u = 0; u < n; ++u) {
        for (auto [v, w] : g.neighbors(u)) {
            random_value = prng_next_int(rng, MIN_RAND_VALUE, MAX_RAND_VALUE);
            g1.add_edge(inv_perms[0][u], inv_perms[1][v], (0.5 * w + random_value) * values[0][u] / values[1][v]);
            g2.add_edge(inv_perms[2][u], inv_perms[3][v], (0.5 * w - random_value) * values[2][u] / values[3][v]);
            edge_set.insert({u, v});
        }
    }

    int m = g1.num_edges();
    int rm = static_cast<int>(m * r);
    for (int i = 0; i < rm; ++i) {
        int u, v;
        do {
            u = rand() % n;
            v = rand() % n;
            if (edge_set.count({u, v}) == 0) {
                random_value = prng_next_int(rng, MIN_RAND_VALUE, MAX_RAND_VALUE);
                g1.add_edge(inv_perms[0][u], inv_perms[1][v], 1.0 * random_value * values[0][u] / values[1][v]);
                g2.add_edge(inv_perms[2][u], inv_perms[3][v], -1.0 * random_value * values[2][u] / values[3][v]);
                edge_set.insert({u, v});
                break;
            }
        } while (1);
    }
}