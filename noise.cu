#include "noise.hpp"
#include "display.hpp"
#include "myutils.hpp"
#include <helper_math.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

static constexpr auto INVSQRT2 = .70710678118654752440;
static constexpr auto N_GRADIENTS = 8;

__device__ int _hash[] = {
	151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
	140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
	247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
	 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
	 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
	 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
	 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
	200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
	 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
	207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
	119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
	129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
	218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
	 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
	184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
	222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
};

__device__ float gradientX[N_GRADIENTS] = {
	1, -1, 0, 0, INVSQRT2, -INVSQRT2, INVSQRT2, -INVSQRT2
};

__device__ float gradientY[N_GRADIENTS] = {
	0, 0, 1, -1, INVSQRT2, INVSQRT2, -INVSQRT2, -INVSQRT2
};

__inline__ __device__ float smooth(float t) {
	return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

/*
 * @param x Grid coordinate x
 * @param y Grid coordinate y (top-left = (0, 0))
 */
__device__ float noiseAt(float x, float y, int seed) {
	
	// Get top-left corner indices
	const int ix = static_cast<int>(x),
	          iy = static_cast<int>(y);

	// Weights
	const float wx = x - ix,
	            wy = y - iy;

	// Get gradients at cell corners
	const int ix0 = ix & 255,
	          iy0 = iy & 255;
	const int ix1 = (ix0 + 1) & 255,
	          iy1 = (iy0 + 1) & 255;
	const int h0 = _hash[ix0],
	          h1 = _hash[ix1];
	const int iTL = (_hash[h0 + iy0] + seed) % N_GRADIENTS,
	          iTR = (_hash[h1 + iy0] + seed) % N_GRADIENTS,
	          iBL = (_hash[h0 + iy1] + seed) % N_GRADIENTS,
	          iBR = (_hash[h1 + iy1] + seed) % N_GRADIENTS;
	const float2 gTopLeft  = make_float2(gradientX[iTL], gradientY[iTL]);
	const float2 gTopRight = make_float2(gradientX[iTR], gradientY[iTR]);
	const float2 gBotLeft  = make_float2(gradientX[iBL], gradientY[iBL]);
	const float2 gBotRight = make_float2(gradientX[iBR], gradientY[iBR]);

	// Calculate dots between distance and gradient vectors
	const float dTopLeft  = dot(gTopLeft,  make_float2(wx,     wy));
	const float dTopRight = dot(gTopRight, make_float2(wx - 1, wy));
	const float dBotLeft  = dot(gBotLeft,  make_float2(wx,     wy - 1));
	const float dBotRight = dot(gBotRight, make_float2(wx - 1, wy - 1));

	const float tx = smooth(wx),
	            ty = smooth(wy);

	const float leftInterp  = lerp(dTopLeft, dBotLeft, ty);
	const float rightInterp = lerp(dTopRight, dBotRight, ty);

	return (lerp(leftInterp, rightInterp, tx) + 1.0) * 0.5;
}

__device__ float sumOctaves(float x, float y, NoiseParams params) {
	float frequency = 1;
	float sum = noiseAt(x * frequency , y * frequency, params.seed);
	float amplitude = 1;
	float range = 1;
	for (int i = 1; i < params.octaves; i++) {
		frequency *= params.lacunarity;
		amplitude *= params.persistence;
		range += amplitude;
		sum += amplitude * noiseAt(x * frequency, y * frequency, params.seed);
	}
	return sum / range;
}

__global__ void perlin(int yStart, int height, NoiseParams params, uint8_t *outPixels) {
	// Pixel coordinates
	const auto px = CUID(x);
	const auto py = CUID(y) + yStart;

	if (px >= Displayer::WIN_WIDTH || py >= yStart + height)
		return;

	auto noise = sumOctaves(px / params.ppu, py / params.ppu, params);

	// Convert noise to pixel
	const auto baseIdx = 4 * LIN(px, py, Displayer::WIN_WIDTH);

	const auto val = noise * 255;

	outPixels[baseIdx + 0] = val;
	outPixels[baseIdx + 1] = val;
	outPixels[baseIdx + 2] = val;
	outPixels[baseIdx + 3] = 255;
}

Stats Perlin::calculate(uint8_t *hPixels, NoiseParams params, cudaStream_t *streams, int nStreams) {
	
	const auto partialHeight = Displayer::WIN_HEIGHT / nStreams;
	const dim3 threads(32, 32);
	const dim3 blocks(std::ceil(Displayer::WIN_WIDTH / 32.0), std::ceil(partialHeight / 32.0));

	cudaEvent_t start, endMalloc, endKernel, endMemcpy;
	MUST(cudaEventCreate(&start));
	MUST(cudaEventCreate(&endMalloc));
	MUST(cudaEventCreate(&endKernel));
	MUST(cudaEventCreate(&endMemcpy));

	MUST(cudaEventRecord(start));
	MUST(cudaMalloc(&dPixels, sizeof(uint8_t) * 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT));
	MUST(cudaEventRecord(endMalloc));

	std::cout << "threads = " << threads << ", blocks = " << blocks << " (= " << threads.x * blocks.x * threads.y * blocks.y << ")" << std::endl;

	for (int i = 0; i < nStreams; ++i) {
		perlin<<<threads, blocks, 0, streams[i]>>>(partialHeight * i, partialHeight, params, dPixels);
	}
	MUST(cudaDeviceSynchronize());
	MUST(cudaEventRecord(endKernel));

	MUST(cudaMemcpy(hPixels, dPixels, sizeof(uint8_t) * 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT,
				cudaMemcpyDeviceToHost));
	MUST(cudaEventRecord(endMemcpy));

	MUST(cudaEventSynchronize(start));
	MUST(cudaEventSynchronize(endMalloc));
	MUST(cudaEventSynchronize(endKernel));
	MUST(cudaEventSynchronize(endMemcpy));

	// Collect stats
	float tMalloc, tKernel, tMemcpy;
	MUST(cudaEventElapsedTime(&tMalloc, start, endMalloc));
	MUST(cudaEventElapsedTime(&tKernel, start, endKernel));
	MUST(cudaEventElapsedTime(&tMemcpy, start, endMemcpy));
	Stats stats;
	stats.tMalloc = tMalloc;
	stats.tKernel = tKernel - tMalloc;
	stats.tMemcpy = tMemcpy - tMalloc - tKernel;
	stats.tTotal = tMemcpy;
	
	// Cleanup
	MUST(cudaFree(dPixels));

	MUST(cudaEventDestroy(endMemcpy));
	MUST(cudaEventDestroy(endKernel));
	MUST(cudaEventDestroy(endMalloc));
	MUST(cudaEventDestroy(start));

	return stats;
}
