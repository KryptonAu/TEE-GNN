#include "csprng.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

struct demo_snapshot {
    uint64_t stream_id;
    uint8_t bytes[32];
    uint32_t u32_value;
    uint64_t u64_value;
    double unit_double;
    double half_open_double;
    double closed_double;
};

static void print_hex(const uint8_t *data, size_t len)
{
    size_t i;

    for (i = 0; i < len; ++i) {
        printf("%02x", data[i]);
    }
}

static int same_double_bits(double a, double b)
{
    return memcmp(&a, &b, sizeof(double)) == 0;
}

static int capture_stream_sample(const csprng_master_t *master,
                                 uint64_t stream_id,
                                 const char *label,
                                 struct demo_snapshot *snapshot)
{
    csprng_stream_t stream;
    int rc;

    memset(&stream, 0, sizeof(stream));
    memset(snapshot, 0, sizeof(*snapshot));

    rc = csprng_stream_init(&stream, master, stream_id, label, strlen(label));
    if (rc != CSPRNG_OK) {
        return rc;
    }

    snapshot->stream_id = stream_id;

    rc = csprng_bytes(&stream, snapshot->bytes, sizeof(snapshot->bytes));
    if (rc != CSPRNG_OK) {
        csprng_stream_wipe(&stream);
        return rc;
    }

    rc = csprng_u32_range(&stream, 100U, 1000U, &snapshot->u32_value);
    if (rc != CSPRNG_OK) {
        csprng_stream_wipe(&stream);
        return rc;
    }

    rc = csprng_u64_range(&stream, 1000000000ULL, 1000000000000ULL, &snapshot->u64_value);
    if (rc != CSPRNG_OK) {
        csprng_stream_wipe(&stream);
        return rc;
    }

    rc = csprng_double(&stream, &snapshot->unit_double);
    if (rc != CSPRNG_OK) {
        csprng_stream_wipe(&stream);
        return rc;
    }

    rc = csprng_double_range(&stream, -3.5, 7.25, &snapshot->half_open_double);
    if (rc != CSPRNG_OK) {
        csprng_stream_wipe(&stream);
        return rc;
    }

    rc = csprng_double_range_inclusive(&stream, 10.0, 11.0, &snapshot->closed_double);
    csprng_stream_wipe(&stream);
    return rc;
}

static void print_snapshot(const struct demo_snapshot *snapshot)
{
    printf("stream %" PRIu64 "\n", snapshot->stream_id);
    printf("  bytes(32)      : ");
    print_hex(snapshot->bytes, sizeof(snapshot->bytes));
    printf("\n");
    printf("  uint32 [100,1000)        : %" PRIu32 "\n", snapshot->u32_value);
    printf("  uint64 [1e9,1e12)        : %" PRIu64 "\n", snapshot->u64_value);
    printf("  double [0,1)             : %.17f\n", snapshot->unit_double);
    printf("  double [-3.5,7.25)       : %.17f\n", snapshot->half_open_double);
    printf("  double [10.0,11.0]       : %.17f\n", snapshot->closed_double);
}

static int snapshots_equal(const struct demo_snapshot *a, const struct demo_snapshot *b)
{
    return a->stream_id == b->stream_id &&
           memcmp(a->bytes, b->bytes, sizeof(a->bytes)) == 0 &&
           a->u32_value == b->u32_value &&
           a->u64_value == b->u64_value &&
           same_double_bits(a->unit_double, b->unit_double) &&
           same_double_bits(a->half_open_double, b->half_open_double) &&
           same_double_bits(a->closed_double, b->closed_double);
}

int main(void)
{
    static const uint8_t master_seed[32] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0x13, 0x57, 0x9b, 0xdf, 0x24, 0x68, 0xac, 0xf0
    };
    const uint64_t stream_ids[] = { 0ULL, 1ULL, 42ULL };
    struct demo_snapshot first_pass[3];
    struct demo_snapshot second_pass[3];
    csprng_master_t master_a;
    csprng_master_t master_b;
    size_t i;
    int rc;

    memset(&master_a, 0, sizeof(master_a));
    memset(&master_b, 0, sizeof(master_b));

    rc = csprng_master_init(&master_a, master_seed, sizeof(master_seed));
    if (rc != CSPRNG_OK) {
        fprintf(stderr, "csprng_master_init(master_a) failed: %d\n", rc);
        return 1;
    }

    rc = csprng_master_init(&master_b, master_seed, sizeof(master_seed));
    if (rc != CSPRNG_OK) {
        fprintf(stderr, "csprng_master_init(master_b) failed: %d\n", rc);
        csprng_master_wipe(&master_a);
        return 1;
    }

    puts("first pass");
    for (i = 0; i < sizeof(stream_ids) / sizeof(stream_ids[0]); ++i) {
        rc = capture_stream_sample(&master_a, stream_ids[i], "demo", &first_pass[i]);
        if (rc != CSPRNG_OK) {
            fprintf(stderr, "capture_stream_sample(first pass, %" PRIu64 ") failed: %d\n",
                    stream_ids[i],
                    rc);
            csprng_master_wipe(&master_a);
            csprng_master_wipe(&master_b);
            return 1;
        }
        print_snapshot(&first_pass[i]);
    }

    puts("");
    puts("second pass after re-initialization");
    for (i = 0; i < sizeof(stream_ids) / sizeof(stream_ids[0]); ++i) {
        rc = capture_stream_sample(&master_b, stream_ids[i], "demo", &second_pass[i]);
        if (rc != CSPRNG_OK) {
            fprintf(stderr, "capture_stream_sample(second pass, %" PRIu64 ") failed: %d\n",
                    stream_ids[i],
                    rc);
            csprng_master_wipe(&master_a);
            csprng_master_wipe(&master_b);
            return 1;
        }
        print_snapshot(&second_pass[i]);
    }

    puts("");
    for (i = 0; i < sizeof(stream_ids) / sizeof(stream_ids[0]); ++i) {
        if (!snapshots_equal(&first_pass[i], &second_pass[i])) {
            fprintf(stderr, "reproducibility check failed for stream %" PRIu64 "\n", stream_ids[i]);
            csprng_master_wipe(&master_a);
            csprng_master_wipe(&master_b);
            return 1;
        }
    }

    if (memcmp(first_pass[0].bytes, first_pass[1].bytes, sizeof(first_pass[0].bytes)) == 0 ||
        memcmp(first_pass[0].bytes, first_pass[2].bytes, sizeof(first_pass[0].bytes)) == 0 ||
        memcmp(first_pass[1].bytes, first_pass[2].bytes, sizeof(first_pass[0].bytes)) == 0) {
        fprintf(stderr, "stream separation check failed: two streams matched unexpectedly\n");
        csprng_master_wipe(&master_a);
        csprng_master_wipe(&master_b);
        return 1;
    }

    puts("reproducibility check: OK");
    puts("stream separation check: OK");

    csprng_master_wipe(&master_a);
    csprng_master_wipe(&master_b);
    return 0;
}
