CXX ?= g++
CC ?= cc

TARGET := test_sdim
BUILD_DIR := .build_test

ROOT_DIR := ..
COMMON_INC := include
CSPRNG_DIR := $(ROOT_DIR)/CSPRNG
EIGEN_INC ?= /usr/include/eigen3

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic
CPPFLAGS += -I$(COMMON_INC) -I$(CSPRNG_DIR) -I$(EIGEN_INC)

CXX_SRCS := \
	test.cpp \
	src/masks.cpp \
	src/csprng_adapter.cpp \
	src/graph.cpp

C_SRCS := \
	$(CSPRNG_DIR)/csprng.c

CXX_OBJS := $(addprefix $(BUILD_DIR)/,$(CXX_SRCS:.cpp=.o))
C_OBJS := $(BUILD_DIR)/csprng.o
OBJS := $(CXX_OBJS) $(C_OBJS)
DEPS := $(OBJS:.o=.d)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)/src
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/csprng.o: $(CSPRNG_DIR)/csprng.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/src: | $(BUILD_DIR)
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) $(BUILD_DIR)

-include $(DEPS)
