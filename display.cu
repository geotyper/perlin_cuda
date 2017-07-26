#include "display.hpp"
#include "myutils.hpp"
#include <cstring>

sf::View keep_ratio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize);

PerlinDisplayer::PerlinDisplayer(World world, float step)
	: world(world)
	, step(step)
	, winWidth(static_cast<int>(world.width/step))
	, winHeight(static_cast<int>(world.height/step))
	, NOISE_MEM_SIZE(winWidth * winHeight * sizeof(float))
	, window(sf::VideoMode(winWidth, winHeight), "Perlin Noise")
{
	window.setFramerateLimit(60);
	hNoise = new float[NOISE_MEM_SIZE/sizeof(float)];
	pixels = new uint8_t[winWidth * winHeight * 4];
	std::cout << "winsize = " << winWidth << " x " << winHeight << std::endl;
}

PerlinDisplayer::~PerlinDisplayer() {
	/*MUST(cudaFreeHost(hNoise));*/
	/*MUST(cudaFreeHost(pixels));*/
	delete[] hNoise;
	delete[] pixels;
}

void PerlinDisplayer::copyData(float *dNoise) {
	std::cout << "NOISE_MEM_SIZE = " << NOISE_MEM_SIZE << std::endl;
	MUST(cudaMemcpy(hNoise, dNoise, NOISE_MEM_SIZE, cudaMemcpyDeviceToHost));
	float maxNoise = 0;
	for (int i = 0; i < NOISE_MEM_SIZE/sizeof(float); ++i)
		if (hNoise[i] > maxNoise)
			maxNoise = hNoise[i];

	#pragma omp parallel for
	for (int i = 0; i < winWidth * winHeight; ++i) {
		if (i < NOISE_MEM_SIZE/sizeof(float)) { // FIXME TODO
			auto p = static_cast<uint8_t>(255 * hNoise[i] / maxNoise);
			printf("hNoise[%d] = %f, p[%d]=%d\n", i, hNoise[i], i, p);
			pixels[4 * i + 0] = p;
			pixels[4 * i + 1] = p;
			pixels[4 * i + 2] = p;
			pixels[4 * i + 3] = p;
		}
	}
}

void PerlinDisplayer::display(float *dNoise) {
	copyData(dNoise);

	sf::Texture tex;
	tex.create(winWidth, winHeight);
	sf::Sprite sprite(tex);
	tex.update(pixels);

	while (window.isOpen()) {
		eventLoop();
		window.clear();
		window.draw(sprite);
		window.display();
	}
}

void PerlinDisplayer::eventLoop() {
	sf::Event evt;
	while (window.pollEvent(evt)) {
		switch (evt.type) {
		case sf::Event::Closed:
			window.close();
			break;
		case sf::Event::Resized:
			window.setView(keep_ratio(evt.size, sf::Vector2u(winWidth, winHeight)));
			break;
		case sf::Event::KeyPressed:
			switch (evt.key.code) {
			case sf::Keyboard::Q:
			case sf::Keyboard::Escape:
				window.close();
				break;
			default:
				break;
			}
		default:
			break;
		}
	}
}

// Handle resizing
sf::View keep_ratio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize) {
	sf::FloatRect viewport(0.f, 0.f, 1.f, 1.f);

	const float screenwidth = size.width / static_cast<float>(designedsize.x),
	            screenheight = size.height / static_cast<float>(designedsize.y);

	if (screenwidth > screenheight) {
		viewport.width = screenheight / screenwidth;
		viewport.left = (1.f - viewport.width) / 2.f;
	} else if (screenwidth < screenheight) {
		viewport.height = screenwidth / screenheight;
		viewport.top = (1.f - viewport.height) / 2.f;
	}

	sf::View view(sf::FloatRect(0, 0, designedsize.x , designedsize.y));
	view.setViewport(viewport);

	return view;
}
