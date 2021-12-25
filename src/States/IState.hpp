#pragma once

#include "../TetrisLogic/Tetris.hpp"
#include "../TetrisVisual/TetrisView.hpp"

namespace pt
{
	class IState
	{
	public:
		virtual void draw() {};
		virtual void update(float) {};
		virtual void eventCheck(sf::Event) {};
		virtual ~IState() {}
	};
}