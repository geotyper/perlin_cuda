#pragma once

#include <cstdint>

struct NoiseParams {
	float ppu;
	int seed;
	int octaves;
	float lacunarity;
	float persistence;
};

class Perlin final {
	uint8_t *dPixels;

public:
	Perlin();
	~Perlin();

	void calculate(uint8_t *hPixels, NoiseParams params, cudaStream_t *streams, int nStreams);
};
