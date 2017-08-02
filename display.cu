#include "display.hpp"
#include "myutils.hpp"
#include <cstring>
#include <iostream>

using std::cout;

Displayer::Displayer()
	: window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Perlin Noise")
{
	tex.create(WIN_WIDTH, WIN_HEIGHT);
}

Displayer::~Displayer() {}

void Displayer::update(uint8_t *hPixels) {
	tex.update(hPixels);
}

void Displayer::draw(std::initializer_list<sf::Drawable*> toDraw) {
	sf::Sprite sprite(tex);
	// hPixels contains RGBA values for the texture's pixels

	window.clear();
	window.draw(sprite);
	for (auto d : toDraw)
		window.draw(*d);
	window.display();
}

bool Displayer::handleEvent(sf::Event evt) {
	_shouldRedraw = false;

	switch (evt.type) {
	case sf::Event::Closed:
		window.close();
		return true;
	case sf::Event::Resized:
		window.setView(keepRatio(evt.size, sf::Vector2u(WIN_WIDTH, WIN_HEIGHT)));
		_shouldRedraw = true;
		return true;
	case sf::Event::KeyPressed:
		switch (evt.key.code) {
		case sf::Keyboard::Q:
		case sf::Keyboard::Escape:
			window.close();
			return true;
		default:
			break;
		}
	default:
		break;
	}
	return false;
}

// Handle resizing
sf::View Displayer::keepRatio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize) {
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
