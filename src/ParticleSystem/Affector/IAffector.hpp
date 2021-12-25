#pragma once
#include "../Particle.hpp"

namespace ps
{
	class IAffector
	{
	public:
		virtual ~IAffector() {};
		virtual IAffector* clone() = 0;
		virtual void update(std::vector<Particle>& particles, float dt) = 0;
	};
}