/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

#include "teegnn_ta.h"

#define TA_UUID TEEGNN_TA_UUID

#define TA_FLAGS (TA_FLAG_SINGLE_INSTANCE | TA_FLAG_MULTI_SESSION)
#define TA_STACK_SIZE (8 * 1024)
#define TA_DATA_SIZE (8 * 1024 * 1024)

#define TA_DESCRIPTION "TEE-GNN inference skeleton TA"
#define TA_CURRENT_TA_EXT_PROPERTIES \
    { "org.linaro.optee.examples.teegnn.property.version", USER_TA_PROP_TYPE_STRING, "0.1" }

#endif /* USER_TA_HEADER_DEFINES_H */

