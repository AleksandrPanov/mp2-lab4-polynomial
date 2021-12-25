#pragma once

#include <SFML/Graphics.hpp>

#include "../../extern/box2d/include/box2d/b2_body.h"
#include "../TetrisLogic/Figure/Figures.hpp"
#include "../TetrisLogic/Figure/FigureCubesBody.hpp"
#include "Vector2f.hpp"

const float radInDeg = 180.f / 3.14f;

namespace pt
{
	class IFigureView : public sf::Drawable
	{
	public:
		virtual sf::Vector2f getPosition() = 0;
		virtual int getId() = 0;
	};

	class FigureCubesView : public IFigureView
	{
	public:
		FigureCubesView(FigureCubesBody& figure, sf::Texture& texture, float scale) :
			m_figure(&figure),
			m_bodies(figure.m_bodies),
			m_texture(texture),
			m_scale(scale)
		{
			Vector2f size(figure.m_cubeSize * m_scale);
			Vector2f offset(size.x / 2);

			sf::Vector2i texSize;
			texSize.x = texture.getSize().x;
			texSize.y = texture.getSize().y;
			texSize.x /= 8;

			sf::Vector2i texPos;
			texPos.x = texSize.x * (rand() % 8);

			sf::Vector2i realTexSize(16, 16);

			m_cube.setOutlineThickness(1.f);
			m_cube.setOutlineColor(sf::Color::Black);

			m_cube.setSize(size);
			m_cube.setOrigin(offset);
			m_cube.setTexture(&m_texture);
			m_cube.setTextureRect(sf::IntRect(texPos, realTexSize));
		}

		void draw(sf::RenderTarget& rt, const sf::RenderStates rs) const override
		{
			for (int i = 0; i < m_bodies.size(); ++i)
			{
				auto& body = m_bodies[i];
				Vector2f pos = body->GetPosition();
				float angleRad = body->GetAngle();

				m_cube.setPosition(sf::Vector2f(pos) * m_scale);
				m_cube.setRotation(angleRad * radInDeg);
				rt.draw(m_cube);
			}
		}

		sf::Vector2f getPosition()
		{
			return sf::Vector2f(Vector2f(m_figure->getPosition())) * m_scale;
		}

		b2Body* getFirstBody()
		{
			return m_bodies[0];
		}

	private:
		FigureCubesBody* m_figure;
		std::vector<b2Body*> m_bodies;
		float m_scale;
		sf::Texture m_texture;
		mutable sf::RectangleShape m_cube;
	};



	class FigureShapeView : public IFigureView
	{
	public:
		FigureShapeView(FigureShapeBody& figure, sf::Texture& texture, float scale) :
			m_figure(&figure),
			m_body(figure.m_body),
			m_texture(texture),
			m_scale(scale)
		{
			Vector2f size(figure.m_cubeSize * m_scale * 0.97f);
			Vector2f offset(size.x / 2);

			sf::Vector2i texSize;
			texSize.x = texture.getSize().x;
			texSize.y = texture.getSize().y;
			texSize.x /= 8;

			sf::Vector2i texPos;
			texPos.x = texSize.x * (rand() % 8);

			sf::Vector2i realTexSize(16, 16);

			m_cube.setOutlineThickness(1.f);
			m_cube.setOutlineColor(sf::Color::Black);

			m_cube.setSize(size);
			m_cube.setTexture(&m_texture);
			m_cube.setTextureRect(sf::IntRect(texPos, realTexSize));
		}

		void draw(sf::RenderTarget& rt, const sf::RenderStates rs) const override
		{
			float angleRad = m_body->GetAngle();
			m_cube.setRotation(angleRad * radInDeg);

			for (auto fixture = m_body->GetFixtureList(); fixture; fixture = fixture->GetNext())
			{
				auto shape = dynamic_cast<b2PolygonShape*>(fixture->GetShape());
				b2Vec2 newOff, & offset = shape->m_vertices[0];
				newOff.x = offset.x * std::cos(angleRad) - offset.y * std::sin(angleRad);
				newOff.y = offset.x * std::sin(angleRad) + offset.y * std::cos(angleRad);
				Vector2f pos = newOff + m_body->GetPosition();

				m_cube.setPosition(sf::Vector2f(pos) * m_scale);
				rt.draw(m_cube, rs);
			}
		}

		sf::Vector2f getPosition() override
		{
			return sf::Vector2f(Vector2f(m_body->GetPosition())) * m_scale;
		}

		int getId() override
		{
			return m_figure->getId();
		}

	private:
		sf::Texture m_texture;
		FigureShapeBody* m_figure;
		float m_scale;
		b2Body* m_body;
		mutable sf::RectangleShape m_cube;
	};
}