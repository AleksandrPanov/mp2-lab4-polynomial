#pragma once
#include <SFML/System/Vector2.hpp>

#include "../extern/box2d/include/box2d/b2_math.h"

namespace pt
{
	struct Vector2f
	{
	public:
		Vector2f(const sf::Vector2f vec = sf::Vector2f())
			: x(vec.x), y(vec.y)
		{ }

		Vector2f(const b2Vec2 vec)
			: x(vec.x), y(vec.y)
		{ }

		explicit Vector2f(float x)
			: x(x), y(x)
		{ }

		Vector2f(float x, float y)
			: x(x), y(y)
		{ }

		operator b2Vec2()       { return b2Vec2(x, y); }
		operator sf::Vector2f() { return sf::Vector2f(x, y); }

	public:
		float x, y;
	};
}