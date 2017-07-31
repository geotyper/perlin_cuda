#pragma once

#include <array>
#include <cstdint>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include "common.hpp"

class Displayer final : public EventHandler {
public:
	static constexpr size_t WIN_WIDTH = 1920,
	                        WIN_HEIGHT = 1080;

private:
	sf::RenderWindow window;
	sf::Texture tex;

	bool _shouldRedraw = false;

	sf::View keepRatio(const sf::Event::SizeEvent& size, const sf::Vector2u& designedsize);

public:
	Displayer();
	~Displayer();

	bool handleEvent(sf::Event event) override;

	void update(uint8_t *hPixels);
	void draw();
	bool shouldRedraw() const { return _shouldRedraw; }
	
	sf::RenderWindow& getWindow() { return window; }
};
