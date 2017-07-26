#include "noise.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include "myutils.hpp"

__global__ static void calcCosines(Grid grid, int n) {
	const auto idx = CUID(x);
	if (idx >= n) return;
	const auto s = grid.x[idx];
	grid.y[idx] = std::sqrt(1.f - s * s);
}

// Creates a grid (width x height), fills it with random gradients and returns it
Grid fillGrid(size_t width, size_t height, unsigned long long seed) {
	// memHalfSize is the size of grid.x (or grid.y)
	const auto size = width * height;
	const auto memHalfSize = sizeof(float) * size;

	// Allocate device memory for the grid
	Grid dGrid;
	MUST(cudaMalloc(&dGrid.x, memHalfSize));
	MUST(cudaMalloc(&dGrid.y, memHalfSize));

	// Generate gradients
	curandGenerator_t dRng;
	MUST_CRND(curandCreateGenerator(&dRng, CURAND_RNG_PSEUDO_DEFAULT));
	MUST_CRND(curandSetPseudoRandomGeneratorSeed(dRng, seed));

	// Generate sines with random values
	MUST_CRND(curandGenerateUniform(dRng, dGrid.x, size));

	// Calculate corresponding cosines
	calcCosines<<<nBlocks(size), nThreads(size)>>>(dGrid, size);

	MUST_CRND(curandDestroyGenerator(dRng));

	return dGrid;
}

void freeGrid(Grid grid) {
	MUST(cudaFree(grid.x));
	MUST(cudaFree(grid.y));
}

__device__ static float dotGridGradient(World world, Grid grid, int ix, int iy, float x, float y) {
	return (grid.x[LIN(ix, iy, static_cast<int>(world.width))] * x +
		grid.y[LIN(ix, iy, static_cast<int>(world.width))] * y);
}

__device__ static float interp(float a, float b, float weight) {
	return (1.f - weight) * a + weight * b;
}

__device__ static float noiseAtPoint(World world, Grid grid, float x, float y) {
	// Find grid cell
	const auto ix = static_cast<int>(x);
	const auto iy = static_cast<int>(y);
	
	// Calculate distance with each corner
	const auto d0 = dotGridGradient(world, grid, ix, iy, x, y);
	const auto d1 = dotGridGradient(world, grid, (ix + 1) % static_cast<int>(world.width), iy, x, y);
	const auto d2 = dotGridGradient(world, grid, ix, (iy + 1) % static_cast<int>(world.height), x, y);
	const auto d3 = dotGridGradient(world, grid, (ix + 1) % static_cast<int>(world.width),
					(iy + 1) % static_cast<int>(world.height), x, y);
	
	// Interpolate
	const auto wx = x - ix; // weight x
	const auto wy = y - iy; // weight y

	const auto i0 = interp(d0, d1, wx);
	const auto i1 = interp(d2, d3, wx);

	return interp(i0, i1, wy);
}

// Calculate the noise at each world point (1 point per thread)
__global__ void perlin(World world, Grid grid, float step, float *result) {
	const auto idx = CUID(x);
	const auto idy = CUID(y);
	printf("idx = %d, idy = %d\n", idx, idy);
	// World coordinates
	const auto x = idx * step;
	const auto y = idy * step;

	/*printf("x = %f, y = %f / wx = %f, wy = %f\n", x, y, world.width, world.height);*/
	if (x > world.width or y > world.height) {
		printf("Exceeeeeeeeded\n");
		return;
	}

	const auto noise = noiseAtPoint(world, grid, x, y);
	
	printf("noise(%f, %f) = %f\n", x, y, noise);
	/*if (x == int(x) and y == int(y))*/
		/*result[LIN(idx, idy, static_cast<size_t>(world.width / step))] = pow(grid.x[LIN(int(x), int(y), int(world.width))], 2) + pow(grid.y[LIN(int(x), int(y), int(world.width))], 2);*/
	/*else*/
		/*result[LIN(idx, idy, static_cast<size_t>(world.width / step))] = 0;//noise;*/
	result[LIN(idx, idy, static_cast<size_t>(world.width / step))] = noise;
}
