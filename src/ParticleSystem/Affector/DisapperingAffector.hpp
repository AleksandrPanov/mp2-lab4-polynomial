#pragma once
#include <SFML/Graphics.hpp>
#include "IAffector.hpp"

namespace ps
{
	class DisapperingAffector : public IAffector
	{
	public:

	private:
		virtual void update(std::vector<Particle>& particles, float dt) override
		{
			for (auto& particle : particles)
			{
				particle.getColor().a = particle.getRemainLiveTime() / particle.getLiveTime() * 255.f;
			}
		}

		IAffector* clone() override
		{
			return new DisapperingAffector();
		}
	};
}