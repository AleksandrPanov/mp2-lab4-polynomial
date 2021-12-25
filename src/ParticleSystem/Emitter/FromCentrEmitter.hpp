#pragma once

#include <SFML/Graphics.hpp>

#include "IEmitter.hpp"

namespace ps
{
	class FromCentrEmittor : public IEmitter
	{
	public:
		FromCentrEmittor() = default;

	private:
		bool isFirst = true;

		virtual void update(std::vector<Particle>& particles, float dt) override
		{
			if (isFirst)
			{
				isFirst = false;
				for (auto& particle : particles)
				{
					particle.setPosition(getRandomCirclePosition(40.f) + getPosition());
				}
			}
			else
			{
				for (auto& particle : particles)
				{
					sf::Vector2f diff = (particle.getPosition() - getPosition());
					diff = diff == sf::Vector2f() ? sf::Vector2f(-1.f, 0.f) : diff;
					diff = diff / sqrt((diff.x * diff.x + diff.y * diff.y));

					float koef = (diff.x * diff.x + diff.y * diff.y) / 100.f;

					koef = 1.f / (koef == 0.f ? 0.001f : koef);
					koef = koef < 5.f ? 5.f : koef;

					particle.setLinearSpeed(diff * koef);
				}
			}
		}

		sf::Vector2f getRandomCirclePosition(int radius)
		{
			float angle = rand() % 628 / 100.f;
			
			sf::Vector2f pos;
			pos.x = cos(angle);
			pos.y = sin(angle);

			return pos * static_cast<float>(rand() % radius);
		}

		IEmitter* clone() override
		{
			return new FromCentrEmittor();
		}
	};
}
