#pragma once

#include <SFML/Graphics.hpp>

class EventHandler {
public:
	virtual bool handleEvent(sf::Event evt) = 0;
};
