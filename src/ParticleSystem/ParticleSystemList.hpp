#pragma once

#include <list>
#include <SFML/Graphics/CircleShape.hpp>

#include "ParticleSystem.hpp"

namespace ps
{
	class ParticleSystemList : public sf::Drawable
	{
	public:
		ParticleSystemList() = default;

		//void addParticleSystem(sf::Vector2f position, const sf::CircleShape& exampleShape, float liveTime, int count)
		//{
		//	m_systems.push_back(std::make_unique<ParticleSystem>(new ParticleSystem(position, exampleShape, liveTime, count)));
		//}

		void addParticleSystem(const ParticleSystemConfig& config)
		{
			ParticleSystem* system = new ParticleSystem(config);
			m_systems.push_back(std::unique_ptr<ParticleSystem>(system));
			system->finished.attach(&ParticleSystemList::onFinishParticleSystem, this);
		}

		void update(float dt)
		{
			deleteFinishedSystems();

			for (auto& system : m_systems)
				system->update(dt);
		}

	private:
		std::list<std::shared_ptr<ParticleSystem>> m_systems;
		std::list<Finishable*> m_finishedSystems;

		void draw(sf::RenderTarget& rt, sf::RenderStates rs) const override
		{
			for (auto& system : m_systems)
				rt.draw(*system, rs);
		}

		void onFinishParticleSystem(Finishable* system)
		{
			m_finishedSystems.push_back(system);
		}

		void deleteFinishedSystems()
		{
			for (auto& finished : m_finishedSystems)
			{
				for (auto it = m_systems.begin(); it != m_systems.end(); ++it)
				{
					if (finished == it->get())
					{
						m_systems.erase(it);
						break;
					}
				}
			}
			m_finishedSystems.clear();
		}
	};
}