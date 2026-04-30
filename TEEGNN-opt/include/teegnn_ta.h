/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef TEEGNN_TA_H
#define TEEGNN_TA_H

/*
 * UUID: 8d1c5322-4a7d-4a3d-9c5f-6f7d31d2b8e1
 */
#define TEEGNN_TA_UUID \
    { 0x8d1c5322, 0x4a7d, 0x4a3d, \
      { 0x9c, 0x5f, 0x6f, 0x7d, 0x31, 0xd2, 0xb8, 0xe1 } }

#define TEEGNN_CMD_INIT_CONTEXT        0
#define TEEGNN_CMD_REMASK              1
#define TEEGNN_CMD_APPLY_ACTIVATION    2
#define TEEGNN_CMD_FINALIZE_RESULT     3
#define TEEGNN_CMD_GET_DEBUG_INFO      4

#endif /* TEEGNN_TA_H */

