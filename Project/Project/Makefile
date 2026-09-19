CC       = g++
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm -lclblast

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
	INCFLAGS += -I../../helper_lib/flow-code
	INCFLAGS += -I../../helper_lib/flow-code/imageLib
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../../helper_lib/helper_lib.a main.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

run: solution
	./solution Dataset/other-data-gray/RubberWhale/frame07.png Dataset/other-data-gray/RubberWhale/frame08.png Dataset/0/output.raw output.png

clean: 
	rm -f solution
	cd ../../helper_lib; make clean