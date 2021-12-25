#pragma once

#include "Finishable.hpp"

namespace ps
{
	class Particle : public Finishable, public sf::Drawable, public sf::Transformable
	{
	public:
		Particle() = default;

		Particle(sf::Vector2f pos, std::shared_ptr<sf::Shape> shape, float liveTime, sf::Color color = sf::Color::White) :
			m_shape(shape),
			m_color(color),
			m_liveTime(liveTime),
			m_remainLiveTime(liveTime),
			m_angularForce(0.f),
			m_angularSpeed(0.f)
		{	}
		
		void update(float dt)
		{
			if ((m_remainLiveTime -= dt) <= 0.f)
				finished(this);

			m_angularSpeed += m_angularForce * dt;
			rotate(m_angularSpeed * dt);

			m_linearSpeed += m_linearForce * dt;
			move(m_linearSpeed * dt);

			m_linearForce = sf::Vector2f(0.f, 0.f);
			m_angularForce = 0;
		}

		void applyForce(sf::Vector2f lin, float ang)
		{
			m_linearForce += lin;
			m_angularForce += ang;
		}

		void setLinearSpeed(sf::Vector2f speed)
		{
			m_linearSpeed = speed;
		}

		void setAngularSpeed(float speed)
		{
			m_angularSpeed = speed;
		}

		void setColor(sf::Color color)
		{
			m_color = color;
		}

		sf::Color& getColor()
		{
			return m_color;
		}

		float getLiveTime()
		{
			return m_liveTime;
		}

		float getRemainLiveTime()
		{
			return m_remainLiveTime;
		}

	private:
		std::shared_ptr<sf::Shape> m_shape;
		sf::Color m_color;
		float m_liveTime;
		float m_remainLiveTime;
		
		sf::Vector2f m_linearForce;
		sf::Vector2f m_linearSpeed;
		float m_angularSpeed;
		float m_angularForce;

		void draw(sf::RenderTarget& rt, sf::RenderStates rs) const override
		{
			m_shape->setFillColor(m_color);
			rt.draw(*m_shape, getTransform());
		}
	};
}