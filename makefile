CC = nvcc
CFLAGS = -std=c++11 --compiler-options -Wall --compiler-options -Wextra --compiler-options -ggdb --compiler-options -fopenmp
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lcurand

all: perlin

perlin: perlin.o noise.o display.o
	$(CC) $(CFLAGS) $^ -o $@.x $(LDFLAGS)

%.o: %.cu myutils.hpp noise.hpp
	$(CC) $(CFLAGS) $< -c

.PHONY: clean
clean:
	rm -f *.x *.o
