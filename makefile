NVCC = nvcc
CC = g++
INCS = -I/opt/cuda/samples/common/inc
NVCFLAGS = -arch sm_30 -std=c++11 --compiler-options -Wall --compiler-options -Wextra --compiler-options -ggdb --compiler-options -Ofast
CFLAGS = -std=c++14 -Wall -Wextra -Ofast
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lpthread
NVLDFLAGS =

all: perlin perlin_seq

perlin: perlin.o display.o noise.o input.o
	$(NVCC) $(INCS) $(NVCFLAGS) $^ -o $@.x $(LDFLAGS) $(NVLDFLAGS)

perlin_seq: perlin_seq.o display.o noise_seq.o input.o
	$(CC) $^ -o $@.x $(LDFLAGS)

noise.o: noise.cu myutils.hpp display.hpp noise.hpp input.hpp
	$(NVCC) $(INCS) $(NVCFLAGS) $< -c

perlin.o: perlin.cu myutils.hpp display.hpp noise.hpp input.hpp
	$(NVCC) $(INCS) $(NVCFLAGS) $< -c

%.o: %.cpp display.hpp noise_seq.hpp input.hpp
	$(CC) $< -c

.PHONY: clean
clean:
	rm -f *.x *.o
