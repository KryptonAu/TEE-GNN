#include "csprng.h"

#include <string.h>

#define CSPRNG_MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * The implementation is fully deterministic:
 * 1. master_seed -> HKDF-Extract -> master PRK
 * 2. (master PRK, fixed domain, stream_id, label) -> HKDF-Expand -> stream key + nonce
 * 3. ChaCha20(key, nonce, counter) -> deterministic byte stream
 *
 * Therefore the same master_seed + stream_id + label + call order always yields
 * identical output, and different stream_id values land in separated sub-streams.
 */

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bit_len;
    uint8_t buffer[64];
    size_t buffer_len;
};

struct hmac_sha256_ctx {
    struct sha256_ctx inner;
    struct sha256_ctx outer;
};

static const uint32_t sha256_k[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static const uint8_t csprng_master_salt[] =
    "TEE-GNN deterministic CSPRNG v1 master salt";
static const uint8_t csprng_stream_domain[] =
    "TEE-GNN deterministic CSPRNG v1 stream";

static uint32_t rotl32(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32U - n));
}

static uint32_t rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32U - n));
}

static uint32_t load32_le(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t load64_le(const uint8_t *p)
{
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}

static void store32_le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xffU);
    p[1] = (uint8_t)((v >> 8) & 0xffU);
    p[2] = (uint8_t)((v >> 16) & 0xffU);
    p[3] = (uint8_t)((v >> 24) & 0xffU);
}

static void store64_le(uint8_t *p, uint64_t v)
{
    p[0] = (uint8_t)(v & 0xffU);
    p[1] = (uint8_t)((v >> 8) & 0xffU);
    p[2] = (uint8_t)((v >> 16) & 0xffU);
    p[3] = (uint8_t)((v >> 24) & 0xffU);
    p[4] = (uint8_t)((v >> 32) & 0xffU);
    p[5] = (uint8_t)((v >> 40) & 0xffU);
    p[6] = (uint8_t)((v >> 48) & 0xffU);
    p[7] = (uint8_t)((v >> 56) & 0xffU);
}

static void secure_bzero(void *ptr, size_t len)
{
    volatile uint8_t *p = (volatile uint8_t *)ptr;

    while (len-- > 0U) {
        *p++ = 0U;
    }
}

static uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

static uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

static uint32_t sha256_bs0(uint32_t x)
{
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

static uint32_t sha256_bs1(uint32_t x)
{
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

static uint32_t sha256_ss0(uint32_t x)
{
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

static uint32_t sha256_ss1(uint32_t x)
{
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

static void sha256_transform(struct sha256_ctx *ctx, const uint8_t block[64])
{
    uint32_t w[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    uint32_t t1;
    uint32_t t2;
    size_t i;

    for (i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    for (i = 16; i < 64; ++i) {
        w[i] = sha256_ss1(w[i - 2]) + w[i - 7] + sha256_ss0(w[i - 15]) + w[i - 16];
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + sha256_bs1(e) + sha256_ch(e, f, g) + sha256_k[i] + w[i];
        t2 = sha256_bs0(a) + sha256_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(struct sha256_ctx *ctx)
{
    ctx->state[0] = 0x6a09e667U;
    ctx->state[1] = 0xbb67ae85U;
    ctx->state[2] = 0x3c6ef372U;
    ctx->state[3] = 0xa54ff53aU;
    ctx->state[4] = 0x510e527fU;
    ctx->state[5] = 0x9b05688cU;
    ctx->state[6] = 0x1f83d9abU;
    ctx->state[7] = 0x5be0cd19U;
    ctx->bit_len = 0U;
    ctx->buffer_len = 0U;
}

static void sha256_update(struct sha256_ctx *ctx, const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    size_t take;

    while (len > 0U) {
        take = CSPRNG_MIN(sizeof(ctx->buffer) - ctx->buffer_len, len);
        memcpy(ctx->buffer + ctx->buffer_len, bytes, take);
        ctx->buffer_len += take;
        ctx->bit_len += (uint64_t)take * 8U;
        bytes += take;
        len -= take;

        if (ctx->buffer_len == sizeof(ctx->buffer)) {
            sha256_transform(ctx, ctx->buffer);
            ctx->buffer_len = 0U;
        }
    }
}

static void sha256_final(struct sha256_ctx *ctx, uint8_t out[32])
{
    size_t i;

    ctx->buffer[ctx->buffer_len++] = 0x80U;

    if (ctx->buffer_len > 56U) {
        while (ctx->buffer_len < 64U) {
            ctx->buffer[ctx->buffer_len++] = 0U;
        }
        sha256_transform(ctx, ctx->buffer);
        ctx->buffer_len = 0U;
    }

    while (ctx->buffer_len < 56U) {
        ctx->buffer[ctx->buffer_len++] = 0U;
    }

    for (i = 0; i < 8; ++i) {
        ctx->buffer[56U + i] = (uint8_t)(ctx->bit_len >> ((7U - i) * 8U));
    }

    sha256_transform(ctx, ctx->buffer);

    for (i = 0; i < 8; ++i) {
        out[i * 4] = (uint8_t)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }

    secure_bzero(ctx, sizeof(*ctx));
}

static void hmac_sha256_init(struct hmac_sha256_ctx *ctx, const uint8_t *key, size_t key_len)
{
    uint8_t key_block[64];
    uint8_t ipad[64];
    uint8_t opad[64];
    uint8_t key_hash[32];
    size_t i;

    memset(key_block, 0, sizeof(key_block));

    if (key_len > sizeof(key_block)) {
        struct sha256_ctx hash_ctx;

        sha256_init(&hash_ctx);
        sha256_update(&hash_ctx, key, key_len);
        sha256_final(&hash_ctx, key_hash);
        memcpy(key_block, key_hash, sizeof(key_hash));
        secure_bzero(key_hash, sizeof(key_hash));
    } else if (key_len > 0U) {
        memcpy(key_block, key, key_len);
    }

    for (i = 0; i < sizeof(key_block); ++i) {
        ipad[i] = (uint8_t)(key_block[i] ^ 0x36U);
        opad[i] = (uint8_t)(key_block[i] ^ 0x5cU);
    }

    sha256_init(&ctx->inner);
    sha256_update(&ctx->inner, ipad, sizeof(ipad));

    sha256_init(&ctx->outer);
    sha256_update(&ctx->outer, opad, sizeof(opad));

    secure_bzero(key_block, sizeof(key_block));
    secure_bzero(ipad, sizeof(ipad));
    secure_bzero(opad, sizeof(opad));
}

static void hmac_sha256_update(struct hmac_sha256_ctx *ctx, const void *data, size_t len)
{
    sha256_update(&ctx->inner, data, len);
}

static void hmac_sha256_final(struct hmac_sha256_ctx *ctx, uint8_t out[32])
{
    uint8_t inner_hash[32];

    sha256_final(&ctx->inner, inner_hash);
    sha256_update(&ctx->outer, inner_hash, sizeof(inner_hash));
    sha256_final(&ctx->outer, out);
    secure_bzero(inner_hash, sizeof(inner_hash));
}

static void hmac_sha256(const uint8_t *key,
                        size_t key_len,
                        const void *data,
                        size_t data_len,
                        uint8_t out[32])
{
    struct hmac_sha256_ctx ctx;

    hmac_sha256_init(&ctx, key, key_len);
    hmac_sha256_update(&ctx, data, data_len);
    hmac_sha256_final(&ctx, out);
    secure_bzero(&ctx, sizeof(ctx));
}

static void hkdf_extract_sha256(const uint8_t *salt,
                                size_t salt_len,
                                const uint8_t *ikm,
                                size_t ikm_len,
                                uint8_t prk[32])
{
    hmac_sha256(salt, salt_len, ikm, ikm_len, prk);
}

static void hkdf_expand_stream_sha256(const uint8_t prk[32],
                                      uint64_t stream_id,
                                      const void *label,
                                      size_t label_len,
                                      uint8_t *okm,
                                      size_t okm_len)
{
    uint8_t header[sizeof(csprng_stream_domain) - 1U + 8U + 4U];
    uint8_t prev[32];
    uint8_t counter = 1U;
    size_t generated = 0U;
    size_t prev_len = 0U;

    memcpy(header, csprng_stream_domain, sizeof(csprng_stream_domain) - 1U);
    store64_le(header + sizeof(csprng_stream_domain) - 1U, stream_id);
    store32_le(header + sizeof(csprng_stream_domain) - 1U + 8U, (uint32_t)label_len);

    while (generated < okm_len) {
        struct hmac_sha256_ctx ctx;
        size_t chunk_len = CSPRNG_MIN((size_t)32U, okm_len - generated);

        hmac_sha256_init(&ctx, prk, 32U);
        if (prev_len > 0U) {
            hmac_sha256_update(&ctx, prev, prev_len);
        }
        hmac_sha256_update(&ctx, header, sizeof(header));
        if (label_len > 0U) {
            hmac_sha256_update(&ctx, label, label_len);
        }
        hmac_sha256_update(&ctx, &counter, 1U);
        hmac_sha256_final(&ctx, prev);
        secure_bzero(&ctx, sizeof(ctx));

        memcpy(okm + generated, prev, chunk_len);
        generated += chunk_len;
        prev_len = sizeof(prev);
        ++counter;
    }

    secure_bzero(header, sizeof(header));
    secure_bzero(prev, sizeof(prev));
}

static void chacha20_quarter_round(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
    *a += *b;
    *d ^= *a;
    *d = rotl32(*d, 16U);

    *c += *d;
    *b ^= *c;
    *b = rotl32(*b, 12U);

    *a += *b;
    *d ^= *a;
    *d = rotl32(*d, 8U);

    *c += *d;
    *b ^= *c;
    *b = rotl32(*b, 7U);
}

static void chacha20_block(const uint8_t key[32],
                           const uint8_t nonce[12],
                           uint32_t counter,
                           uint8_t out[64])
{
    static const uint32_t constants[4] = {
        0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U
    };
    uint32_t state[16];
    uint32_t working[16];
    size_t i;

    state[0] = constants[0];
    state[1] = constants[1];
    state[2] = constants[2];
    state[3] = constants[3];

    for (i = 0; i < 8; ++i) {
        state[4U + i] = load32_le(key + i * 4U);
    }

    state[12] = counter;
    state[13] = load32_le(nonce);
    state[14] = load32_le(nonce + 4U);
    state[15] = load32_le(nonce + 8U);

    memcpy(working, state, sizeof(working));

    for (i = 0; i < 10; ++i) {
        chacha20_quarter_round(&working[0], &working[4], &working[8], &working[12]);
        chacha20_quarter_round(&working[1], &working[5], &working[9], &working[13]);
        chacha20_quarter_round(&working[2], &working[6], &working[10], &working[14]);
        chacha20_quarter_round(&working[3], &working[7], &working[11], &working[15]);

        chacha20_quarter_round(&working[0], &working[5], &working[10], &working[15]);
        chacha20_quarter_round(&working[1], &working[6], &working[11], &working[12]);
        chacha20_quarter_round(&working[2], &working[7], &working[8], &working[13]);
        chacha20_quarter_round(&working[3], &working[4], &working[9], &working[14]);
    }

    for (i = 0; i < 16; ++i) {
        working[i] += state[i];
        store32_le(out + i * 4U, working[i]);
    }

    secure_bzero(state, sizeof(state));
    secure_bzero(working, sizeof(working));
}

static int csprng_stream_refill(csprng_stream_t *stream)
{
    if (stream == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    if (stream->exhausted) {
        return CSPRNG_ERR_LIMIT;
    }

    chacha20_block(stream->key, stream->nonce, stream->block_counter, stream->buffer);

    if (stream->block_counter == UINT32_MAX) {
        stream->exhausted = 1;
    } else {
        ++stream->block_counter;
    }

    stream->buffer_pos = 0U;
    return CSPRNG_OK;
}

int csprng_master_init(csprng_master_t *master, const void *seed, size_t seed_len)
{
    if (master == NULL || seed == NULL || seed_len == 0U) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    hkdf_extract_sha256(csprng_master_salt,
                        sizeof(csprng_master_salt) - 1U,
                        (const uint8_t *)seed,
                        seed_len,
                        master->prk);
    return CSPRNG_OK;
}

int csprng_stream_init(csprng_stream_t *stream,
                       const csprng_master_t *master,
                       uint64_t stream_id,
                       const void *label,
                       size_t label_len)
{
    uint8_t material[CSPRNG_CHACHA20_KEY_SIZE + CSPRNG_CHACHA20_NONCE_SIZE];

    if (stream == NULL || master == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    if (label == NULL && label_len != 0U) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    hkdf_expand_stream_sha256(master->prk, stream_id, label, label_len, material, sizeof(material));

    memcpy(stream->key, material, CSPRNG_CHACHA20_KEY_SIZE);
    memcpy(stream->nonce,
           material + CSPRNG_CHACHA20_KEY_SIZE,
           CSPRNG_CHACHA20_NONCE_SIZE);
    stream->block_counter = 0U;
    stream->buffer_pos = CSPRNG_CHACHA20_BLOCK_SIZE;
    stream->exhausted = 0;
    memset(stream->buffer, 0, sizeof(stream->buffer));

    secure_bzero(material, sizeof(material));
    return CSPRNG_OK;
}

int csprng_bytes(csprng_stream_t *stream, void *out, size_t out_len)
{
    uint8_t *dst = (uint8_t *)out;
    size_t take;
    int rc;

    if (stream == NULL || (out == NULL && out_len != 0U)) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    while (out_len > 0U) {
        if (stream->buffer_pos == CSPRNG_CHACHA20_BLOCK_SIZE) {
            rc = csprng_stream_refill(stream);
            if (rc != CSPRNG_OK) {
                return rc;
            }
        }

        take = CSPRNG_MIN(CSPRNG_CHACHA20_BLOCK_SIZE - stream->buffer_pos, out_len);
        memcpy(dst, stream->buffer + stream->buffer_pos, take);
        stream->buffer_pos += take;
        dst += take;
        out_len -= take;
    }

    return CSPRNG_OK;
}

int csprng_u32(csprng_stream_t *stream, uint32_t *out)
{
    uint8_t tmp[4];
    int rc;

    if (out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    rc = csprng_bytes(stream, tmp, sizeof(tmp));
    if (rc != CSPRNG_OK) {
        return rc;
    }

    *out = load32_le(tmp);
    return CSPRNG_OK;
}

int csprng_u64(csprng_stream_t *stream, uint64_t *out)
{
    uint8_t tmp[8];
    int rc;

    if (out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    rc = csprng_bytes(stream, tmp, sizeof(tmp));
    if (rc != CSPRNG_OK) {
        return rc;
    }

    *out = load64_le(tmp);
    return CSPRNG_OK;
}

int csprng_u32_range(csprng_stream_t *stream,
                     uint32_t min_inclusive,
                     uint32_t max_exclusive,
                     uint32_t *out)
{
    uint32_t span;
    uint32_t threshold;
    uint32_t x;
    int rc;

    if (out == NULL || max_exclusive <= min_inclusive) {
        return CSPRNG_ERR_RANGE;
    }

    span = max_exclusive - min_inclusive;
    threshold = (uint32_t)(0U - span) % span;

    /*
     * Do not use x % span directly on raw random bits: when 2^32 is not a
     * multiple of span, some residues would appear slightly more often.
     */
    do {
        rc = csprng_u32(stream, &x);
        if (rc != CSPRNG_OK) {
            return rc;
        }
    } while (x < threshold);

    *out = min_inclusive + (x % span);
    return CSPRNG_OK;
}

int csprng_u64_range(csprng_stream_t *stream,
                     uint64_t min_inclusive,
                     uint64_t max_exclusive,
                     uint64_t *out)
{
    uint64_t span;
    uint64_t threshold;
    uint64_t x;
    int rc;

    if (out == NULL || max_exclusive <= min_inclusive) {
        return CSPRNG_ERR_RANGE;
    }

    span = max_exclusive - min_inclusive;
    threshold = (uint64_t)(0ULL - span) % span;

    do {
        rc = csprng_u64(stream, &x);
        if (rc != CSPRNG_OK) {
            return rc;
        }
    } while (x < threshold);

    *out = min_inclusive + (x % span);
    return CSPRNG_OK;
}

int csprng_double(csprng_stream_t *stream, double *out)
{
    uint64_t x;
    int rc;

    if (out == NULL) {
        return CSPRNG_ERR_BAD_PARAMS;
    }

    rc = csprng_u64(stream, &x);
    if (rc != CSPRNG_OK) {
        return rc;
    }

    /*
     * A binary64 has 53 bits of precision. Using the top 53 random bits gives
     * a uniform discrete value in [0, 1) over exactly representable steps.
     */
    *out = (double)(x >> 11) * (1.0 / 9007199254740992.0);
    return CSPRNG_OK;
}

int csprng_double_range(csprng_stream_t *stream, double a, double b, double *out)
{
    double u;
    int rc;

    if (out == NULL || !(a < b)) {
        return CSPRNG_ERR_RANGE;
    }

    rc = csprng_double(stream, &u);
    if (rc != CSPRNG_OK) {
        return rc;
    }

    *out = a + (b - a) * u;
    return CSPRNG_OK;
}

int csprng_double_range_inclusive(csprng_stream_t *stream, double a, double b, double *out)
{
    uint64_t x;
    double u;
    int rc;

    if (out == NULL || !(a <= b)) {
        return CSPRNG_ERR_RANGE;
    }

    if (a == b) {
        *out = a;
        return CSPRNG_OK;
    }

    rc = csprng_u64(stream, &x);
    if (rc != CSPRNG_OK) {
        return rc;
    }

    /*
     * This variant maps 53 random bits to [0, 1] so the upper end can be hit.
     * The distribution is still deterministic but remains discrete at double
     * precision, which matches binary64 semantics.
     */
    u = (double)(x >> 11) * (1.0 / 9007199254740991.0);
    *out = a + (b - a) * u;
    return CSPRNG_OK;
}

void csprng_master_wipe(csprng_master_t *master)
{
    if (master != NULL) {
        secure_bzero(master, sizeof(*master));
    }
}

void csprng_stream_wipe(csprng_stream_t *stream)
{
    if (stream != NULL) {
        secure_bzero(stream, sizeof(*stream));
    }
}
