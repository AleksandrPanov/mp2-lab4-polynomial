#pragma once

#include "../Particle.hpp"

namespace ps
{
	class IEmitter : public sf::Transformable
	{
	public:
		virtual ~IEmitter() {};
		virtual IEmitter* clone() = 0;
		virtual void update(std::vector<Particle>& particles, float dt) = 0;
	};
}