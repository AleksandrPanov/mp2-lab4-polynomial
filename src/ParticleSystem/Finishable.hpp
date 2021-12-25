#pragma once

#include "../Event.hpp"

namespace ps
{
	class Finishable
	{
	public:
		event<void(Finishable*)> finished;
		virtual ~Finishable() {};
	};
}