#pragma once
#include "States.hpp"
#include "../Resources.hpp"

namespace pt
{
	InGameState::InGameState(Scene* controller) :
		m_scene(controller),
		m_tetris(controller->getTetris()),
		m_view(controller->getView())
	{
		m_tetris.gameOver.attach(&InGameState::onGameOver, this);
		m_tetris.setCurrentFigureSpeed(m_tetris.normalFigureSpeed);
		Resources::getInstance().BackMusic.setLoop(true);
		Resources::getInstance().BackMusic.setVolume(10.f);
		Resources::getInstance().BackMusic.play();
	}

	InGameState::~InGameState()
	{
 		m_tetris.gameOver.detachLater(&InGameState::onGameOver, *this);
	}

	void InGameState::draw()
	{
		m_view.draw();
	}

	void InGameState::update(float dt)
	{
		m_view.update(dt);
		m_tetris.update(dt);
	}

	void InGameState::eventCheck(sf::Event event)
	{
		switch (event.type)
		{
		case sf::Event::KeyPressed: onKeyPressed(event); break;
		case sf::Event::KeyReleased: onKeyReleased(event); break;
		default: break;
		}
	}

	void InGameState::onKeyPressed(sf::Event event)
	{
		switch (event.key.code)
		{
		case sf::Keyboard::Q:
			if (!m_isRightRotateKeyPressed)
			{
				m_tetris.rotateCurrentFigure(-PI / 2.f);
				m_isRightRotateKeyPressed = 1;
			}
			break;
		case sf::Keyboard::E:
			if (!m_isLeftRotateKeyPressed)
			{
				m_tetris.rotateCurrentFigure(PI / 2.f);
				m_isLeftRotateKeyPressed = 1;
			}
			break;
		case sf::Keyboard::D: 
		{
			b2Vec2 offset;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
			{
				offset.x = m_tetris.normalMoveOffset.x * 2.f;
				offset.y = m_tetris.normalMoveOffset.y * 2.f;
			}
			else
			{
				offset = m_tetris.normalMoveOffset;
			}
			m_tetris.moveCurrentFigure(offset);
			break;
		}
		case sf::Keyboard::A:
		{
			b2Vec2 offset;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
			{
				offset.x = -m_tetris.normalMoveOffset.x * 2.f;
				offset.y = -m_tetris.normalMoveOffset.y * 2.f;
			}
			else
			{
				offset = -m_tetris.normalMoveOffset;
			}
			m_tetris.moveCurrentFigure(offset);
			break;
		}
		case sf::Keyboard::S: m_tetris.setCurrentFigureSpeed(m_tetris.fastFigureSpeed); break;
		default: break;
		}
	}

	void InGameState::onKeyReleased(sf::Event event)
	{
		switch (event.key.code)
		{
		case sf::Keyboard::Q: m_isRightRotateKeyPressed = 0; break;
		case sf::Keyboard::E: m_isLeftRotateKeyPressed = 0; break;
		case sf::Keyboard::D: break;
		case sf::Keyboard::A: break;
		case sf::Keyboard::S: m_tetris.setCurrentFigureSpeed(m_tetris.normalFigureSpeed); break;
		default: break;
		}
	}

	void InGameState::onGameOver()
	{
		Resources::getInstance().BackMusic.stop();
		m_scene->setState(new MenuState(m_scene));
	}
}