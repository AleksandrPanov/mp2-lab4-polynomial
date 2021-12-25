#pragma once

#include "Rules.hpp"

namespace pt
{
	StandartRules::StandartRules(Tetris* tetris) :
		m_tetris(tetris)
	{	}

	bool StandartRules::isGameWon()
	{
		return false;
	}

	bool StandartRules::isGameOver()
	{
		return m_tetris->getFallFigureCount() >= 3;
	}
}