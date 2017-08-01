#include "display.hpp"
#include "myutils.hpp"
#include "noise.hpp"
#include <curand.h>
#include <iostream>
#include <array>
#include <unistd.h>

using std::cout;
using std::endl;

int main(int argc, char **argv) {

	float ppu = 100.f;
	int seed = 0;
	if (argc > 1) {
		ppu = atof(argv[1]);
		if (argc > 2)
			seed = atoi(argv[2]);
	}
	Displayer displayer;
	sf::RenderWindow& window = displayer.getWindow();
	Perlin perlin;

	uint8_t *hPixels;
	MUST(cudaMallocHost(&hPixels, 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT * sizeof(uint8_t)));

	constexpr auto N_STREAMS = 4;
	cudaStream_t streams[N_STREAMS];
	for (int i = 0; i < N_STREAMS; ++i) {
		MUST(cudaStreamCreate(&streams[i]));
	}

	perlin.calculate(hPixels, ppu, seed, streams, N_STREAMS);
	/*for (int i = 0; i < 4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT * sizeof(uint8_t); ++i) {*/
		/*cout << "pixel[" << i << "] = " << int(hPixels[i]) << endl;*/
	/*}*/
	displayer.update(hPixels);

	while (window.isOpen()) {
		
		// Event loop
		sf::Event evt;
		while (window.pollEvent(evt)) {
			if (displayer.handleEvent(evt))
				break;
			/*if (input.handleEvent(evt))*/
				/*break;*/
		}

		bool shouldRedraw = displayer.shouldRedraw();

		/*if (input.shouldRecalculate()) {*/
			/*// Recalculate perlin*/

			/*displayer.update(dPixels);*/
			/*shouldRedraw = true;*/
		/*}*/

		if (shouldRedraw) {
			displayer.draw();
		}

		usleep(16666);
	}
}
