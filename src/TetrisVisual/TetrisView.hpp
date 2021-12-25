#pragma once
#include <SFML/Graphics.hpp>
#include <list>

#include "../TetrisLogic/Tetris.hpp"
#include "Event.hpp"

#include "FigureView.hpp"
#include "FloorView.hpp"
#include "FigureHighLight.hpp"

#include "../ParticleSystem/ParticleSystemList.hpp"
#include "ParticlesTemplate.hpp"

#include "../Resources.hpp"

namespace pt
{
	class TetrisView
	{
	public:
		TetrisView(sf::RenderWindow& window, Tetris& m_tetris, float scale) :
			m_window(window),
			m_tetris(m_tetris),
			m_highLight(scale),
			m_scale(scale),
			m_floor(nullptr)
		{ 
			m_tetris.newFigureCreate.attach(&TetrisView::onNewFigureCreate, this);
			m_tetris.figureFall.attach(&TetrisView::onFigureFall, this);
			m_tetris.getSelectedFigure().figureCollide.attach(&TetrisView::onFigureCollide, this);

			m_backroundTexture = Resources::getInstance().BackgroundTex;
			m_backroundTexture.setRepeated(true);

			m_figuresTexture = Resources::getInstance().FigureTex;

			m_floorTexture = Resources::getInstance().FloorTex;
			m_floorTexture.setRepeated(true);

			m_floor.reset(new FloorView(m_tetris.getFloor(), m_floorTexture, scale));

			m_lastView.reset(getDefaultViewRect());
			m_neededView.reset(getDefaultViewRect());
		}

		~TetrisView()
		{
			m_tetris.newFigureCreate.detach(&TetrisView::onNewFigureCreate, this);
			m_tetris.figureFall.detach(&TetrisView::onFigureFall, this);
			m_tetris.getSelectedFigure().figureCollide.detach(&TetrisView::onFigureCollide, this);
		}

		void restart()
		{
			m_figures.clear();
			m_lastView.reset(getDefaultViewRect());
			m_neededView.reset(getDefaultViewRect());
		}

		void update(float dt)
		{
			m_particleSystems.update(dt);
			calculateNeededView();
		}

		void draw()
		{
			// Backround draw
			backgroundDraw();

			// View setting
			m_lastView = getNextView();
			m_window.setView(m_lastView);

			// Hightight draw
			m_window.draw(m_highLight);

			// Figure draw
			for (auto& figure : m_figures)
				m_window.draw(*figure);

			// Floor draw
			m_window.draw(*m_floor);

			// Highest point draw (debug)
			/*sf::CircleShape higherPoint(2.f, 120u);
			higherPoint.setOrigin(sf::Vector2f(1.f, 1.f));
			higherPoint.setPosition(sf::Vector2f(Vector2f(m_tetris.getHigherPosition())) * m_scale);
			higherPoint.setFillColor(sf::Color::Green);
			m_window.draw(higherPoint);*/

			// Partical system draw
			m_window.draw(m_particleSystems);

			// Default wiew set
			m_window.setView(m_window.getDefaultView());
		}

	private:
		sf::RenderWindow& m_window;
		Tetris& m_tetris;
		FigureHighLight m_highLight;
		std::unique_ptr<FloorView> m_floor;
		std::list<std::shared_ptr<IFigureView>> m_figures;

		sf::View m_lastView;
		sf::View m_neededView;

		ps::ParticleSystemList m_particleSystems;

		sf::Texture m_backroundTexture;
		sf::Texture m_floorTexture;
		sf::Texture m_figuresTexture;
		float m_scale;

		void onNewFigureCreate(IFigureBody* figure)
		{
			std::shared_ptr<IFigureView> ptr;
			auto castedFigure = dynamic_cast<FigureShapeBody*>(figure);
			ptr.reset(new FigureShapeView(*castedFigure, m_figuresTexture, m_scale));
			m_figures.push_back(ptr);

			m_highLight.setFigure(figure);

			ps::ParticleSystemConfig config = particles::FigureSpawnParticles;
			config.position = sf::Vector2f(Vector2f(figure->getPosition())) * m_scale;
			m_particleSystems.addParticleSystem(config);

			Resources::getInstance().FigureSpawnSound.play();
		}

		void onFigureCollide(b2Vec2 figurePos)
		{
			ps::ParticleSystemConfig config = particles::FigureCollisionParticles;
			config.position = sf::Vector2f(Vector2f(figurePos)) * m_scale;
			m_particleSystems.addParticleSystem(config);

			//Resources::FigureCollishionSound.play();
		}

		void onFigureFall(IFigureBody*& figure)
		{
			m_figures.remove_if([figure](std::shared_ptr<IFigureView>& checkFigure) -> bool
				{
					return figure->getId() == checkFigure->getId();
				});
		}

		void calculateNeededView()
		{
			auto view = m_neededView;
			auto size = view.getSize();
			auto centr = view.getCenter();

			auto upperSpawnPoint = sf::Vector2i(0, m_window.getSize().y * 0.10f);
			auto lowerSpawnPoint = sf::Vector2i(0, m_window.getSize().y * 0.20f);
			auto figureLowerPoint = sf::Vector2i(0, m_window.getSize().y * 0.80f);

			auto upperLine = m_window.mapPixelToCoords(upperSpawnPoint, m_neededView).y; // spawn point beetwen 10-20% window height
			auto lowerLine = m_window.mapPixelToCoords(lowerSpawnPoint, m_neededView).y;
			auto spawnLine = m_tetris.getSpawnPosition().y * m_scale;
			auto lowerFigureLine = m_window.mapPixelToCoords(figureLowerPoint, m_neededView).y;

			auto curFigureLine = m_tetris.getSelectedFigure().getFigure()->getPosition().y;

			if (spawnLine < upperLine)
			{
				centr.y -= upperLine - spawnLine;
			}
			else if (spawnLine > lowerLine)
			{
				centr.y -= spawnLine - lowerLine;
			}

			m_neededView = sf::View(centr, view.getSize());
		}

		sf::View getNextView()
		{
			auto view = m_lastView;

			constexpr float zoomDelta = 1.f / 10.f;
			constexpr float moveDelta = 1.f / 30.f;

			view.setCenter(view.getCenter() + (m_neededView.getCenter() - view.getCenter()) * moveDelta);
			view.setSize(view.getSize() + (m_neededView.getSize() - view.getSize()) * zoomDelta);

			return view;
		}

		void backgroundDraw()
		{
			sf::Sprite sprite(m_backroundTexture);

			sf::Vector2i leftUpCoord(0, 0);
			sf::Vector2i start(m_window.mapPixelToCoords(leftUpCoord).x, m_window.mapPixelToCoords(leftUpCoord).y);

			sprite.setPosition(start.x, start.y);
			m_window.draw(sprite);
		}

		sf::FloatRect getDefaultViewRect()
		{
			return sf::FloatRect(
				-static_cast<float>(m_window.getSize().x) / 2.f,
				-static_cast<float>(m_window.getSize().y),
				m_window.getSize().x,
				m_window.getSize().y);
		}
	};
}