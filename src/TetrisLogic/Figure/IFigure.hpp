#pragma once

#include "../../../extern/box2d/include/box2d/box2d.h"

namespace pt
{
	class IFigureBody
	{
	public:
		virtual ~IFigureBody() {};

		virtual bool isCollide() = 0;
		virtual void rotate(float angle) = 0;
		virtual void move(b2Vec2 offset) {};

		virtual void setPosition(b2Vec2 pos) = 0;
		virtual void setAngularVelocity(float velocity) = 0;
		virtual void setVelocity(b2Vec2 velocity) = 0;
		virtual void setGravityScale(float scale) = 0;

		virtual b2Vec2 getPosition() = 0;
		virtual float getAngle() = 0;
		virtual int getId() = 0;
		virtual b2AABB getBounds() { return b2AABB(); };
	};
}