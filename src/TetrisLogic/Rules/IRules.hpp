#pragma once

namespace pt
{
	class IRules
	{
	public:
		virtual bool isGameWon() = 0;
		virtual bool isGameOver() = 0;
	};
}