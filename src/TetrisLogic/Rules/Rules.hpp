#pragma once

#include "IRules.hpp"

namespace pt
{
	class Tetris;

	class StandartRules : public IRules
	{
	public:
		StandartRules(Tetris* tetris);

		virtual bool isGameWon() override;
		virtual bool isGameOver() override;

	private:
		Tetris* m_tetris;
	};
}