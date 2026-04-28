TEEC_EXPORT=/data/link/export/usr
CROSS_COMPILE=aarch64-linux-gnu-

CXX 	= g++
CC      = $(CROSS_COMPILE)gcc
LD      = $(CROSS_COMPILE)ld
AR      = $(CROSS_COMPILE)ar
NM      = $(CROSS_COMPILE)nm
OBJCOPY = $(CROSS_COMPILE)objcopy
OBJDUMP = $(CROSS_COMPILE)objdump
READELF = $(CROSS_COMPILE)readelf

TARGET := teegnn_host

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.cpp) teegnn_main.cpp
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

CXXFLAGS += -Wall -Wextra -O2 -fPIC -std=c++17 -I/usr/local/include/eigen3
CXXFLAGS += -I$(INC_DIR) -I../include

ifdef TEEC_EXPORT
CXXFLAGS += -I$(TEEC_EXPORT)/include
LDFLAGS  += -L$(TEEC_EXPORT)/lib
endif

LDLIBS += -lteec

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -o bin/$@ $^ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) bin