#pragma once

#include "../ParticleSystem/ParticleSystem.hpp"
#include "../ParticleSystem/Emitter/FromCentrEmitter.hpp"
#include "../ParticleSystem/Emitter/PhysicalEmitter.hpp"
#include "../ParticleSystem/Affector/DisapperingAffector.hpp"
#include "../ParticleSystem/Affector/RainbowAffector.hpp"

namespace particles
{
	ps::ParticleSystemConfig FigureSpawnParticles = {
		new ps::RainbowAffector(),
		new ps::FromCentrEmittor(),
		new sf::CircleShape(5.f),
		sf::Vector2f(),
		100,
		1.f
	};

	ps::ParticleSystemConfig FigureCollisionParticles = {
		new ps::DisapperingAffector(),
		new ps::PhysicalEmitter(),
		new sf::CircleShape(10.f),
		sf::Vector2f(),
		30,
		2.f
	};
}
