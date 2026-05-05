# SPDX-License-Identifier: BSD-2-Clause

global-incdirs-y += ../include

srcs-y += teegnn_ta.c
srcs-y += crypto_optee.c
srcs-y += ../src/csprng.c
srcs-y += ../src/csprng_adapter_c.c
srcs-y += ../src/scan.c
srcs-y += ../src/blocked_csc.c
srcs-y += ../src/csc_graph.c