#pragma once

#include "../../extern/box2d/include/box2d/box2d.h"
#include "Event.hpp"
#include "Figure/Figures.hpp"

namespace pt
{
	class SelectedFigure
	{
	public:
		event<void(b2Vec2)> figureCollide;

		SelectedFigure(b2World& world, float cubeSize) :
			m_world(world),
			m_figure(nullptr),
			m_cubeSize(cubeSize),
			m_verticalSpeed(0.f, 0.f),
			m_horizontalSpeed(cubeSize * 2.f, 0),
			m_angle(0.f)
		{	}

		void update()
		{
			tryRotate();
			
			tryMove();

			m_figure->setVelocity(m_summarVelocity);
			m_summarVelocity = m_verticalSpeed;

			if (isCollideWithOtherFigures())
			{
				m_figure->setGravityScale(1.f);
				figureCollide(m_figure->getPosition());
			}
		}

		void setFigure(IFigureBody* figure)
		{
			if (m_figure)
			{
				m_figure->setAngularVelocity(0.f);
				m_figure->setGravityScale(1.f);
			}

			m_figure = figure;

			if (figure)
			{
				m_position = m_figure->getPosition();
				m_angle = 0.f;
				m_figure->setGravityScale(0.f);
			}
		}

		void move(b2Vec2 offset)
		{
			m_position += offset;
		}

		void rotate(float angle)
		{
			m_angle += angle;
		}

		IFigureBody* getFigure()
		{
			return m_figure;
		}

		void setSpeed(b2Vec2 speed)
		{
			m_verticalSpeed = speed;
		}

	private:
		b2World& m_world;
		IFigureBody* m_figure;
		b2Vec2 m_verticalSpeed;
		b2Vec2 m_horizontalSpeed;
		b2Vec2 m_position;
		b2Vec2 m_summarVelocity;
		float m_cubeSize;
		float m_angle;
		float m_rotateSpeed = 12.f;

		void tryMove()
		{
			// Smooth move
			if ((m_position.x - m_figure->getPosition().x) > 0.f)
				m_summarVelocity += b2Vec2((m_position.x - m_figure->getPosition().x) / .1f, 0.f);
			else if ((m_position.x - m_figure->getPosition().x) < 0.f)
				m_summarVelocity += b2Vec2((m_position.x - m_figure->getPosition().x) / .1f, 0.f);
			// log
			//std::cout << m_figure->getPosition().x << ' ' << m_figure->getPosition().y << '\n';
		}

		void tryRotate()
		{
			// Smooth rotation
			float eps = 0.001;
			if (m_figure->getAngle() > m_angle - eps && m_figure->getAngle() < m_angle + eps)
				m_figure->setAngularVelocity(0.f);
			else if (m_figure->getAngle() > m_angle - eps)
				m_figure->setAngularVelocity(-m_rotateSpeed * (m_figure->getAngle() - m_angle - eps));
			else if (m_figure->getAngle() < m_angle + eps)
				m_figure->setAngularVelocity(m_rotateSpeed * (m_angle - eps - m_figure->getAngle()));
		}

		bool isCollideWithOtherFigures()
		{
			return m_figure->isCollide();
		}
	};
}
