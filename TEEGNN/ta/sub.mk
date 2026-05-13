# SPDX-License-Identifier: BSD-2-Clause

global-incdirs-y += ../include
global-incdirs-y += include

srcs-y += teegnn_ta.c
srcs-y += src/crypto_optee.c
srcs-y += ../src/csprng.c
srcs-y += src/csprng_adapter_c.c
srcs-y += src/scan.c
srcs-y += ../src/blocked_csc.c
srcs-y += src/csc_graph.c
srcs-y += src/edge_list.c
srcs-y += src/blocked_edge_list.c