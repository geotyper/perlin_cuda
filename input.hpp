#pragma once

#include <SFML/Graphics.hpp>
#include "event_handler.hpp"
#include "noise_params.hpp"

class Input final : public EventHandler, public sf::Drawable {
	NoiseParams& params;
	sf::Font font;
	bool legendEnabled = true;

public:
	Input(NoiseParams& params);

	void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
	bool handleEvent(sf::Event event) override;
};
