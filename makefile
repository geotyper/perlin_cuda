CC = nvcc
INCS = -I/opt/cuda/samples/common/inc
CFLAGS = -arch sm_30 -std=c++11 --compiler-options -Wall --compiler-options -Wextra --compiler-options -ggdb --compiler-options -fopenmp
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lcurand

all: perlin

perlin: perlin.o display.o noise.o input.o
	$(CC) $(INCS) $(CFLAGS) $^ -o $@.x $(LDFLAGS)

%.o: %.cu myutils.hpp display.hpp noise.hpp input.hpp
	$(CC) $(INCS) $(CFLAGS) $< -c

.PHONY: clean
clean:
	rm -f *.x *.o
