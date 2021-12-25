#pragma once

#include "InGameState.hpp"

namespace pt
{
	RestartGameState::RestartGameState(Scene* controller) :
		m_scene(controller)
	{
		controller->getView().restart();
		controller->getTetris().restart();
	}

	void RestartGameState::update(float)
	{
		m_scene->setState(new InGameState(m_scene));
	}

	void RestartGameState::draw()
	{
		m_scene->setState(new InGameState(m_scene));
	}
}