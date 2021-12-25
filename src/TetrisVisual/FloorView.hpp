#pragma once
#include <SFML/Graphics.hpp>
#include "../../extern/box2d/include/box2d/b2_body.h"

#include "../TetrisLogic/Floor.hpp"
#include "Vector2f.hpp"

#define radInDeg 180.f / 3.14f

namespace pt
{
	class FloorView : public sf::Drawable
	{
	public:
		FloorView(Floor& floor, sf::Texture& texture, float scale) :
			m_body(floor.m_body),
			m_texture(texture),
			m_scale(scale)
		{
			sf::Vector2f size(sf::Vector2f(Vector2f(floor.m_size)) * scale);
			sf::Vector2f offset(size.x / 2, size.y / 2);

			m_cube.setTexture(&m_texture);

			m_cube.setSize(size);
			m_cube.setOrigin(offset);
			m_cube.setPosition(sf::Vector2f(Vector2f(m_body->GetPosition())) * scale);
			m_cube.setRotation(m_body->GetAngle() * radInDeg);
		}

		void draw(sf::RenderTarget& rt, const sf::RenderStates rs) const override
		{
			rt.draw(m_cube, rs);
		}

	private:
		float m_scale;
		b2Body* m_body;
		sf::Texture& m_texture;
		sf::RectangleShape m_cube;
	};
}