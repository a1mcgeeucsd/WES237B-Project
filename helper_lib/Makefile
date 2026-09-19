CC       = g++
CFLAGS   = -g -Wall
INCFLAGS :=

FLOW_DIR = ./flow-code
FLOW_INC = -I$(FLOW_DIR) -I$(FLOW_DIR)/imageLib
INCFLAGS += $(FLOW_INC)

ifeq ($(shell uname -o), Darwin)
	LDFLAGS = -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  = -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS = -lOpenCL
endif
LDFLAGS += -lm

# 1. Your original C files
C_SOURCES = device.c kernel.c matrix.c img.c
C_OBJECTS = $(C_SOURCES:.c=.o)

# 2. Middlebury C++ files explicitly defined
FLOW_OBJECTS = flowIO.o RefCntMem.o Image.o

# 3. Combine both sets of object files
OBJECTS = $(C_OBJECTS) $(FLOW_OBJECTS)

.PHONY: all clean debug
all: helper_lib.a

debug: CFLAGS += -DOCL_DEVICE_TYPE=CL_DEVICE_TYPE_CPU
debug: helper_lib.a

# 4. Standard rule for your local C files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $< $(INCFLAGS)

# 5. Explicit rules for Middlebury files to ensure they actually compile
flowIO.o: $(FLOW_DIR)/flowIO.cpp
	$(CC) $(CFLAGS) -c -o $@ $< $(INCFLAGS)

RefCntMem.o: $(FLOW_DIR)/imageLib/RefCntMem.cpp
	$(CC) $(CFLAGS) -c -o $@ $< $(INCFLAGS)

Image.o: $(FLOW_DIR)/imageLib/Image.cpp
	$(CC) $(CFLAGS) -c -o $@ $< $(INCFLAGS)

# 6. Archive all objects together into the static library
helper_lib.a: $(OBJECTS)
	ar rcs $@ $(OBJECTS)

clean: 
	rm -f $(OBJECTS) helper_lib.a
