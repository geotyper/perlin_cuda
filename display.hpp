#pragma once

#include "common.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

class PerlinDisplayer final {
	const World world;
	const float step;
	const int winWidth, winHeight;
	const size_t NOISE_MEM_SIZE;

	sf::RenderWindow window;
	float *hNoise;
	uint8_t *pixels;

	void eventLoop();
	void copyData(float *dNoise);
public:
	PerlinDisplayer(World world, float step);
	~PerlinDisplayer();

	void display(float *dNoise);
};
