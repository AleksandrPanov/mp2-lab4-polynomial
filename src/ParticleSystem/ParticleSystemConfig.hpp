#pragma once

#include "ParticleSystem.hpp"
#include "Emitter/IEmitter.hpp"
#include "Affector/IAffector.hpp"

namespace ps
{
	class ParticleSystemConfig
	{
	public:
		std::unique_ptr<IAffector> affector;
		std::unique_ptr<IEmitter> emitter;
		std::shared_ptr<sf::Shape> shape;
		sf::Vector2f position;
		unsigned int count;
		float liveTime;

		ParticleSystemConfig(IAffector* affector = nullptr, 
			IEmitter* emitter = nullptr,
			sf::Shape* shape = nullptr, 
			sf::Vector2f position = sf::Vector2f(), 
			unsigned int count = 0, float liveTime = .0f) :
			affector(affector),
			emitter(emitter),
			shape(shape),
			position(position),
			count(count),
			liveTime(liveTime)
		{	}
		ParticleSystemConfig(const ParticleSystemConfig& other)
		{
			*this = other;
		}
		ParticleSystemConfig& operator=(const ParticleSystemConfig& other)
		{
			affector = std::unique_ptr<IAffector>(other.affector->clone());
			emitter = std::unique_ptr<IEmitter>(other.emitter->clone());
			shape = other.shape;
			position = other.position;
			count = other.count;
			liveTime = other.liveTime;

			return *this;
		}
	};
}