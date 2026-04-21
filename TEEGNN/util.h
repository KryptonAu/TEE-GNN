#pragma once

#ifdef __cplusplus
#include <cassert>
#include <cstdint>
#include <limits>
#else
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#endif

#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <set>

#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PRNG PRNG;
void prng_seed(PRNG *rng, uint32_t seed);
PRNG prng_make(uint32_t seed);
uint32_t prng_next_u32(PRNG *rng);
int prng_next_int(PRNG *rng, int min_value, int max_value);
double prng_next_double(PRNG *rng);
uint32_t generate_rrng_seed(PRNG *rng);

void random_permutation(PRNG *rng, int n, int *perm);
void random_vector(PRNG *rng, int n, int min_val, int max_val, int *vec);

// Random Permutation Matrix
typedef struct RPM RPM;
void generate_rpm(PRNG *rng, int n, RPM *rpm);

// Subtly Designed Invertible Matrix
typedef struct SDIM SDIM;
void generate_sdim(PRNG *rng, int n, SDIM *sdim);

// Low-rank Mask Matrix
typedef struct LMM LMM;
void generate_lmm(PRNG *rng, int n, int rank, LMM *lmm);

void free_rpm(RPM *rpm);
void free_sdim(SDIM *sdim);
void free_lmm(LMM *lmm);

void weight_mask(double *in, double *out, SDIM *L, SDIM *R);
void feature_mask(double *in, double *out, RPM *L, RPM *R, LMM *M);

#ifdef __cplusplus
}
#endif

void graph_split(PRNG *rng, const Graph &g, Weighted_Graph &g1, Weighted_Graph &g2, 
                 const std::vector<RPM*> &rpms, double r);