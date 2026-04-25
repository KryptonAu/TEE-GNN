// SPDX-License-Identifier: BSD-2-Clause

#include <err.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tee_client_api.h>

#include "teegnn_ta.h"

static void invoke_init(TEEC_Session *sess)
{
    TEEC_Operation op;
    uint32_t origin = 0;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT,
                                     TEEC_NONE,
                                     TEEC_NONE,
                                     TEEC_NONE);
    op.params[0].value.a = 0;
    op.params[0].value.b = 0;

    res = TEEC_InvokeCommand(sess, TEEGNN_CMD_INIT_CONTEXT, &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "INIT_CONTEXT failed: 0x%x origin 0x%x", res, origin);
}

static void invoke_restore(TEEC_Session *sess)
{
    TEEC_Operation op;
    uint32_t origin = 0;
    TEEC_Result res;
    uint8_t y1[8] = { 0 };
    uint8_t y2[8] = { 0 };
    uint8_t out[8] = { 0 };

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_VALUE_INPUT);
    op.params[0].tmpref.buffer = y1;
    op.params[0].tmpref.size = sizeof(y1);
    op.params[1].tmpref.buffer = y2;
    op.params[1].tmpref.size = sizeof(y2);
    op.params[2].tmpref.buffer = out;
    op.params[2].tmpref.size = sizeof(out);

    res = TEEC_InvokeCommand(sess, TEEGNN_CMD_RESTORE_AGGREGATION, &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "RESTORE_AGGREGATION failed: 0x%x origin 0x%x", res, origin);
}

static void invoke_activation(TEEC_Session *sess)
{
    TEEC_Operation op;
    uint32_t origin = 0;
    TEEC_Result res;
    uint8_t activations[8] = { 0 };

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT,
                                     TEEC_VALUE_INPUT,
                                     TEEC_NONE,
                                     TEEC_NONE);
    op.params[0].tmpref.buffer = activations;
    op.params[0].tmpref.size = sizeof(activations);

    res = TEEC_InvokeCommand(sess, TEEGNN_CMD_APPLY_ACTIVATION, &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "APPLY_ACTIVATION failed: 0x%x origin 0x%x", res, origin);
}

static void invoke_finalize(TEEC_Session *sess)
{
    TEEC_Operation op;
    uint32_t origin = 0;
    TEEC_Result res;
    uint8_t logits[8] = { 0 };
    uint8_t labels[8] = { 0 };
    uint8_t pred[8] = { 0 };

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_VALUE_OUTPUT);
    op.params[0].tmpref.buffer = logits;
    op.params[0].tmpref.size = sizeof(logits);
    op.params[1].tmpref.buffer = labels;
    op.params[1].tmpref.size = sizeof(labels);
    op.params[2].tmpref.buffer = pred;
    op.params[2].tmpref.size = sizeof(pred);

    res = TEEC_InvokeCommand(sess, TEEGNN_CMD_FINALIZE_RESULT, &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "FINALIZE_RESULT failed: 0x%x origin 0x%x", res, origin);
}

static void usage(const char *argv0)
{
    fprintf(stderr, "Usage: %s /dataset/cora\n", argv0);
}

int main(int argc, char **argv)
{
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_UUID uuid = TEEGNN_TA_UUID;
    uint32_t origin = 0;
    TEEC_Result res;
    const char *dataset_dir = NULL;

    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }
    dataset_dir = argv[1];

    /*
     * Real REE integration will load graph support, masked tensors, weights, and
     * labels from dataset_dir, then pass TEEC shared memory handles to the TA.
     */
    printf("TEE-GNN host skeleton loading dataset metadata from: %s\n", dataset_dir);

    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InitializeContext failed: 0x%x", res);

    res = TEEC_OpenSession(&ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_OpenSession failed: 0x%x origin 0x%x", res, origin);

    invoke_init(&sess);
    invoke_restore(&sess);
    invoke_activation(&sess);
    invoke_finalize(&sess);

    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
    printf("TEE-GNN OP-TEE skeleton invocation completed.\n");
    return 0;
}
