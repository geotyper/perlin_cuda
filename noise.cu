#include "noise.hpp"
#include "display.hpp"
#include "myutils.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>

static constexpr auto INVSQRT2 = .70710678118654752440;

__device__ float gradientX[] = {
	1, -1, 0, 0, INVSQRT2, -INVSQRT2, INVSQRT2, -INVSQRT2
};

__device__ float gradientY[] = {
	0, 0, 1, -1, INVSQRT2, INVSQRT2, -INVSQRT2, -INVSQRT2
};

/*
 * @param x Grid coordinate x
 * @param y Grid coordinate y (top-left = (0, 0))
 */
__device__ float noiseAt(float x, float y, int seed) {
	return 0;
}

__global__ void perlin(int yStart, int height, int seed, float ppu, uint8_t *outPixels) {
	// Pixel coordinates
	const auto px = CUID(x);
	const auto py = CUID(y) + yStart;

	if (px >= Displayer::WIN_WIDTH || py >= yStart + height)
		return;

	/*const auto noise = noiseAt(px / ppu, py / ppu, seed);*/

	// Convert noise to pixel
	const auto baseIdx = 4 * LIN(px, py, Displayer::WIN_WIDTH);
	/*const auto val = noise * 255;*/
	outPixels[baseIdx + 0] = 255;
	outPixels[baseIdx + 1] = 255;
	outPixels[baseIdx + 2] = 255;
	outPixels[baseIdx + 3] = 255;
}

Perlin::Perlin() {}

Perlin::~Perlin() {}

__global__ void foo(uint8_t *out) {
	const auto px = CUID(x);
	const auto py = CUID(y);

	if (px >= Displayer::WIN_WIDTH || py >= Displayer::WIN_HEIGHT)
		return;

	out[LIN(px, py, Displayer::WIN_WIDTH)] = 255;
	/*out[px] = 255;*/
}

void Perlin::calculate(uint8_t *hPixels, cudaStream_t *streams, int nStreams) {
	
	const auto partialHeight = Displayer::WIN_HEIGHT / nStreams;
	printf("partialHeight = %d\n", partialHeight);
	const dim3 threads(32, 32);
	const dim3 blocks(std::ceil(Displayer::WIN_WIDTH / 32), std::ceil(partialHeight / 32));

	MUST(cudaMalloc(&dPixels, sizeof(uint8_t) * 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT));

	std::cout << "threads = " << threads << ", blocks = " << blocks << " ( = " << threads.x * blocks.x * threads.y * blocks.y << ")" << std::endl;

	for (int i = 0; i < nStreams; ++i) {
		perlin<<<threads, blocks>>>(partialHeight * i, partialHeight, 0, 1, dPixels);
		MUST(cudaMemcpyAsync(hPixels + 4 * partialHeight * i * sizeof(uint8_t),
				dPixels + 4 * partialHeight * i * sizeof(uint8_t),
				4 * sizeof(uint8_t) * partialHeight,
				cudaMemcpyDeviceToHost, streams[i]));
	}

	MUST(cudaFree(dPixels));
}
