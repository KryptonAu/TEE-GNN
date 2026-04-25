# SPDX-License-Identifier: BSD-2-Clause

HOST_APP := teegnn_host

OPTEE_CLIENT_EXPORT ?= /opt/optee_client
TA_DEV_KIT_DIR ?= /opt/optee_os/out/arm/export-ta_arm32
TEEC_EXPORT ?= $(OPTEE_CLIENT_EXPORT)

HOST_CFLAGS += -Wall -Wextra -O2 -I$(TEEC_EXPORT)/include -Ita/include
HOST_LDFLAGS += -L$(TEEC_EXPORT)/lib -lteec

.PHONY: all host ta clean

all: host ta

host:
	$(CC) $(HOST_CFLAGS) -o $(HOST_APP) host/main.c $(HOST_LDFLAGS)

ta:
	$(MAKE) -C ta TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)

clean:
	rm -f $(HOST_APP)
	$(MAKE) -C ta clean TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)

