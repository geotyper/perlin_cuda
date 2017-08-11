#include "display.hpp"
#include "myutils.hpp"
#include "noise.hpp"
#include "input.hpp"
#include <curand.h>
#include <iostream>
#include <array>
#include <unistd.h>

using std::cout;
using std::endl;

int main(int argc, char **argv) {

	NoiseParams params;
	// Default parameters
	params.ppu = 250.f;
	params.seed = 0;
	params.octaves = 3;
	params.lacunarity = 2;
	params.persistence = 0.5;
	int nStreams = 2;
	if (argc > 1) {
		nStreams = std::min(100, atoi(argv[1]));
	}
	std::cout << "Using " << nStreams << " CUDA streams." << std::endl;

	Displayer displayer;
	Input input(params);
	sf::RenderWindow& window = displayer.getWindow();
	Perlin perlin;

	uint8_t *hPixels;
	MUST(cudaMallocHost(&hPixels, 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT * sizeof(uint8_t)));

	cudaStream_t *streams = new cudaStream_t[nStreams];
	for (int i = 0; i < nStreams; ++i) {
		MUST(cudaStreamCreate(&streams[i]));
	}

	auto stats = perlin.calculate(hPixels, params, streams, nStreams);
	/*for (int i = 0; i < 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT * sizeof(uint8_t); ++i) {*/
		/*cout << "pixel[" << i << "] = " << int(hPixels[i]) << endl;*/
	/*}*/
	displayer.update(hPixels);

	std::cout << stats << std::endl;

	while (window.isOpen()) {
		
		// Event loop
		sf::Event evt;
		bool shouldRecalculate = false;
		while (window.pollEvent(evt)) {
			if (displayer.handleEvent(evt))
				break;
			if (input.handleEvent(evt)) {
				shouldRecalculate = true;
				break;
			}
		}

		bool shouldRedraw = displayer.shouldRedraw();

		if (shouldRecalculate) {
			// Recalculate perlin
			stats = perlin.calculate(hPixels, params, streams, nStreams);
			displayer.update(hPixels);
			shouldRedraw = true;
			std::cout << stats << std::endl;
		}

		if (shouldRedraw) {
			displayer.draw({ &input });
		}

		usleep(16666);
	}

	// Cleanup
	for (int i = 0; i < nStreams; ++i) {
		MUST(cudaStreamDestroy(streams[i]));
	}
	MUST(cudaFreeHost(hPixels));
	delete[] streams;
}
