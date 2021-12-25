#pragma once

#include <list>
#include <vector>

#include "Particle.hpp"
#include "ParticleSystemConfig.hpp"

namespace ps
{
	class ParticleSystem : public Finishable, public sf::Drawable
	{
	public:
		ParticleSystem() = default;
		ParticleSystem(const ParticleSystem&) = default;

		ParticleSystem(const ParticleSystemConfig& config) :
			m_config(config),
			m_particles(m_config.count, Particle(m_config.position, m_config.shape, m_config.liveTime))
		{
			for (auto& particle : m_particles)
				particle.finished.attach(&ParticleSystem::onParticleFinished, this); // without detach

			m_config.emitter->setPosition(m_config.position);
		};

		void update(float dt)
		{
			removeDeadParticles();

			m_config.emitter->update(m_particles, dt);
			m_config.affector->update(m_particles, dt);

			for (auto& particle : m_particles)
				particle.update(dt);
		}

	private:
		ParticleSystemConfig m_config;
		std::vector<Particle> m_particles;
		std::list<Finishable*> m_deadParticles;
		std::shared_ptr<sf::Shape> m_shape;

		void draw(sf::RenderTarget& rt, sf::RenderStates rs) const override
		{
			for (auto& particle : m_particles)
				rt.draw(particle, rs);
		}

		void onParticleFinished(Finishable* particle)
		{
			m_deadParticles.push_back(particle);
		}

		void removeDeadParticles()
		{
			for (auto& deadParticle : m_deadParticles)
			{
				for (auto& particle : m_particles)
				{
					if (&particle == deadParticle)
					{
						std::swap(particle, *std::prev(m_particles.end()));
						break;
					}
				}
			}

			m_particles.resize(m_particles.size() - m_deadParticles.size());
			m_deadParticles.clear();

			if (m_particles.empty())
				finished(this);
		}
	};
}