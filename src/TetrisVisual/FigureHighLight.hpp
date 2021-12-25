#pragma once

#include <SFML/Graphics.hpp>
#include "../../extern/box2d/include/box2d/box2d.h"

#include "../TetrisLogic/Figure/Figures.hpp"

namespace pt
{
	class FigureHighLight : public sf::Drawable
	{
	public:
		FigureHighLight(IFigureBody* figure, float scale) :
			m_figure(figure),
			m_scale(scale)
		{	};

		FigureHighLight(float scale) :
			FigureHighLight(nullptr, scale)
		{	};

		void setFigure(IFigureBody* figure)
		{
			m_figure = figure;
		}

	private:
		IFigureBody* m_figure;
		float m_scale;

		void draw(sf::RenderTarget& rt, sf::RenderStates rs) const override
		{
			sf::RectangleShape highlight;
			auto aabb = m_figure->getBounds();

			auto up = rt.mapPixelToCoords(sf::Vector2i(0.f, 0.f));
			auto down = rt.mapPixelToCoords(sf::Vector2i(0.f, rt.getSize().y));
			auto diff = down - up;

			sf::Vector2f highlightSize(
				std::fabs(aabb.lowerBound.x - aabb.upperBound.x) * m_scale,
				diff.y
			);

			sf::Vector2f pos = rt.mapPixelToCoords(sf::Vector2i(0, 0));

			highlight.setPosition(sf::Vector2f(aabb.lowerBound.x * m_scale, pos.y));
			highlight.setSize(highlightSize);
			highlight.setFillColor(sf::Color(255u, 255u, 255u, 70u));
			rt.draw(highlight);
		}
	};
}