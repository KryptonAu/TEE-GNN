CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -pedantic

TARGET := csprng_demo
SRCS := main.c csprng.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c csprng.h
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

# OP-TEE integration:
# Add csprng.c and csprng.h to your TA sources, then compile ta_example.c inside
# an OP-TEE TA build with CSPRNG_WITH_OPTEE_TA defined. The Linux demo does not
# build ta_example.c because this machine has no TA SDK/toolchain.
