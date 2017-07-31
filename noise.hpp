#pragma once

#include <cstdint>

class Perlin final {
	uint8_t *dPixels;

public:
	Perlin();
	~Perlin();

	void calculate(uint8_t *hPixels, cudaStream_t *streams, int nStreams);
};
