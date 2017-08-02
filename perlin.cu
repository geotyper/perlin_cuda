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
	params.ppu = 200.f;
	params.seed = 0;
	params.octaves = 3;
	params.lacunarity = 2;
	params.persistence = 0.5;
	if (argc > 1) {
		params.ppu = atof(argv[1]);
		if (argc > 2)
			params.seed = atoi(argv[2]);
	}

	Displayer displayer;
	Input input(params);
	sf::RenderWindow& window = displayer.getWindow();
	Perlin perlin;

	uint8_t *hPixels;
	MUST(cudaMallocHost(&hPixels, 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT * sizeof(uint8_t)));

	constexpr auto N_STREAMS = 2;
	cudaStream_t streams[N_STREAMS];
	for (int i = 0; i < N_STREAMS; ++i) {
		MUST(cudaStreamCreate(&streams[i]));
	}

	auto stats = perlin.calculate(hPixels, params, streams, N_STREAMS);
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
			stats = perlin.calculate(hPixels, params, streams, N_STREAMS);
			displayer.update(hPixels);
			shouldRedraw = true;
			std::cout << stats << std::endl;
		}

		if (shouldRedraw) {
			displayer.draw({&input});
		}

		usleep(16666);
	}

	// Cleanup
	for (int i = 0; i < N_STREAMS; ++i) {
		MUST(cudaStreamDestroy(streams[i]));
	}
	MUST(cudaFreeHost(hPixels));
}
