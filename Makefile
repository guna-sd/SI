CC = gcc
CFLAGS = -Ofast -march=native -funroll-all-loops -g
LIBS = -lm -fopenmp

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

TARGET = ./build/SI

all: $(BUILD_DIR) $(TARGET) post-build

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

post-build: $(TARGET)
	@echo "Running post setup script..."
	bash setup.sh

clean:
	rm -r $(BUILD_DIR)

.PHONY: all clean post-build
