#include "input.hpp"

Input::Input(NoiseParams& params) : params(params) {
	font.loadFromFile("Hack-Regular.ttf");
}

bool Input::handleEvent(sf::Event evt) {
	switch (evt.type) {
	case sf::Event::KeyPressed:
		switch (evt.key.code) {
		case sf::Keyboard::W:
			params.ppu *= 1.2;
			return true;
		case sf::Keyboard::S:
			params.ppu *= 0.8;
			return true;
		case sf::Keyboard::E:
			params.seed++;
			return true;
		case sf::Keyboard::D:
			params.seed--;
			return true;
		case sf::Keyboard::R:
			params.octaves++;
			return true;
		case sf::Keyboard::F:
			params.octaves--;
			return true;
		case sf::Keyboard::T:
			params.lacunarity += 0.1;
			return true;
		case sf::Keyboard::G:
			params.lacunarity = std::max(0.f, params.lacunarity - 0.1f);
			return true;
		case sf::Keyboard::Y:
			params.persistence += 0.1;
			return true;
		case sf::Keyboard::H:
			params.persistence = std::max(0.f, params.persistence - 0.1f);
			return true;
		case sf::Keyboard::C:
			legendEnabled = !legendEnabled;
			return true;
		default:
			break;
		}
	default:
		break;
	}

	return false;
}

void Input::draw(sf::RenderTarget& target, sf::RenderStates states) const {
	if (!legendEnabled) return;

	sf::Text text("Controls:", font, 30);
	text.setPosition(5, 5);
	target.draw(text, states);
	text.setCharacterSize(24);
	auto a = 25;
	text.setPosition(text.getPosition().x, text.getPosition().y + 2 * a);
	text.setString("W: ppu+         [" + std::to_string(params.ppu) + "]");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("S: ppu-");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("E: seed+        [" + std::to_string(params.seed) + "]");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("D: seed--");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("R: octaves+     [" + std::to_string(params.octaves) + "]");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("F: octaves-");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("T: lacunarity+  [" + std::to_string(params.lacunarity) + "]");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("G: lacunarity-");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("Y: persistence+ [" + std::to_string(params.persistence) + "]");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("H: persistence-");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("C: toggle controls");
	target.draw(text, states);
	text.setPosition(text.getPosition().x, text.getPosition().y + a);
	text.setString("Q: quit");
	target.draw(text, states);
}
