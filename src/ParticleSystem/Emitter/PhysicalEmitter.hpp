#pragma once

#include <SFML/Graphics.hpp>

#include "IEmitter.hpp"

namespace ps
{
	class PhysicalEmitter : public IEmitter
	{
	public:
		PhysicalEmitter() = default;

	private:
		bool isFirst = true;

		virtual void update(std::vector<Particle>& particles, float dt) override
		{
			if (isFirst)
			{
				isFirst = false;
				for (auto& particle : particles)
				{
					particle.setPosition(getRandomRectPosition(30.f, 10.f) + sf::Vector2f(0.f, 10.f) + getPosition());
					particle.setLinearSpeed(particle.getPosition() - getPosition());
				}
			}
			else
			{
				for (auto& particle : particles)
				{
					particle.applyForce(sf::Vector2f(0.f, 20.f), 0.f);
				}
			}
		}

		sf::Vector2f getRandomRectPosition(int width, int height)
		{
			return sf::Vector2f((rand() % (width * 2)) - (width), (rand() % (height * 2)) - (height));
		}

		IEmitter* clone() override
		{
			return new PhysicalEmitter();
		}
	};
}
