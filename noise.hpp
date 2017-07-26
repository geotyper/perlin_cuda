#pragma once

#include "common.hpp"
#include <cstdio>

struct Grid {
	float *x;
	float *y;
};

////// HOST FUNCTIONS

// Creates a grid (width x height), fills it with random gradients and returns it
__host__ Grid fillGrid(size_t width, size_t height, unsigned long long seed);

__host__ void freeGrid(Grid grid);

////// KERNEL FUNCTIONS

// Calculate the noise at each world point (1 point per thread)
__global__ void perlin(World world, Grid grid, float step, float *result);

__global__ inline void shit() {
	printf("ASDSADSA\n");
}

////// DEVICE FUNCTIONS

//__device__ constexpr float dotGridGradient(World world, Grid grid, int ix, int iy, float x, float y);

//__device__ constexpr float interp(float a, float b, float weight);

//__device__ float noiseAtPoint(World world, Grid grid, float x, float y);

