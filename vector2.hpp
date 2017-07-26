#pragma once

#include <random>
#include <ostream>

// Random number generation
std::random_device rng;
std::uniform_real_distribution<float> distribution(0, 1);

struct Vector2 {
	float x;
	float y;
	static Vector2 random() {
		const auto sin = distribution(rng);
		// cos2 = 1 - sin2
		return Vector2 { std::sqrt(1 - sin * sin), sin };
	}
};

std::ostream& operator<<(std::ostream& o, Vector2 v) {
	o << "{ " << v.x << ", " << v.y << " }";
	return o;
}
