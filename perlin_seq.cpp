// Perlin noise generator and visualizer - sequential version
// by G. Parolini && I. Cislaghi
// 2017
#include "display.hpp"
#include "noise_seq.hpp"
#include "input.hpp"
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

	Displayer displayer;
	Input input(params);
	sf::RenderWindow& window = displayer.getWindow();
	PerlinSeq perlin;

	uint8_t *hPixels = new uint8_t[4 * Displayer::WIN_WIDTH * Displayer::WIN_HEIGHT];

	auto stats = perlin.calculate(hPixels, params);
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
			stats = perlin.calculate(hPixels, params);
			displayer.update(hPixels);
			shouldRedraw = true;
			std::cout << stats << std::endl;
		}

		if (shouldRedraw) {
			displayer.draw({ &input });
		}

		usleep(16666);
	}

	delete[] hPixels;
}
