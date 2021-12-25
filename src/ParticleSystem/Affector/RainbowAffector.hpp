#pragma once
#include <SFML/Graphics.hpp>
#include "IAffector.hpp"

namespace ps
{
	class RainbowAffector : public IAffector
	{
	public:

	private:
		bool isFirst = true;

		virtual void update(std::vector<Particle>& particles, float dt) override
		{
			if (isFirst)
			{
				isFirst = false;
				for (auto& particle : particles)
				{
					particle.getColor().r = rand() % 256;
					particle.getColor().g = rand() % 256;
					particle.getColor().b = rand() % 256;
				}
			}
			else
			{
				for (auto& particle : particles)
				{
					/*float intencive = 0.05f;
					int colorType = rand() % 3;
					switch (colorType)
					{
					case 0:
						particle.getColor().r += 255.f * intencive; break;
					case 1:
						particle.getColor().g += 255.f * intencive; break;
					case 2:
						particle.getColor().b += 255.f * intencive; break;
					}*/
					particle.getColor().a = particle.getRemainLiveTime() / particle.getLiveTime() * 255.f;
				}
			}
		}

		IAffector* clone() override
		{
			return new RainbowAffector();
		}
	};
}