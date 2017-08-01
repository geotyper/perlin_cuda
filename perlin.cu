#include "display.hpp"
#include "myutils.hpp"
#include "noise.hpp"
#include <curand.h>
#include <iostream>
#include <array>
#include <unistd.h>

using std::cout;
using std::endl;

int main() {
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

	perlin.calculate(hPixels, streams, N_STREAMS);
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
