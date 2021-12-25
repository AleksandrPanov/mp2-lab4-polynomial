#pragma once

#include "TetrisLogic/Tetris.hpp"
#include "TetrisVisual/TetrisView.hpp"
#include "TetrisVisual/MenuView.hpp"
#include "States/IState.hpp"

namespace pt
{
	class Scene
	{
	public:
		Scene(Tetris& tetris, TetrisView& view, MenuView& menu);

		void setState(IState* state);

		void checkEvent(sf::Event event);
		void draw();
		void update(float dt);

		Tetris& getTetris();
		TetrisView& getView();
		MenuView& getMenu();

	private:
		Tetris& m_tetris;
		TetrisView& m_view;
		MenuView& m_menu;
		std::unique_ptr<IState> m_currentState;
	};

	Scene::Scene(Tetris& tetris, TetrisView& view, MenuView& menu) :
		m_tetris(tetris),
		m_view(view),
		m_menu(menu),
		m_currentState(nullptr)
	{	}

	void Scene::setState(IState* state)
	{
		m_currentState.reset(state);
	}

	void Scene::checkEvent(sf::Event event)
	{
		m_currentState->eventCheck(event);
	}

	void Scene::draw()
	{
		m_currentState->draw();
	}

	void Scene::update(float dt)
	{
		m_currentState->update(dt);
	}

	Tetris& Scene::getTetris()
	{
		return m_tetris;
	}
	TetrisView& Scene::getView()
	{
		return m_view;
	}
	MenuView& Scene::getMenu()
	{
		return m_menu;
	}
}