#pragma once
#include "../../extern/box2d/include/box2d/box2d.h"

namespace pt
{
	class Floor
	{
	public:
		Floor(b2World& world, b2Vec2 position, b2Vec2 size) :
			m_position(position),
			m_size(size)
		{
			b2BodyDef floorDef;
			b2PolygonShape polygon;
			b2FixtureDef floorFixtureDef;

			floorDef.type = b2BodyType::b2_staticBody;
			floorDef.position = position;

			polygon.SetAsBox(size.x / 2, size.y / 2);
			floorFixtureDef.shape = &polygon;
			floorFixtureDef.restitution = 0;
			floorFixtureDef.restitutionThreshold = 0;
			floorFixtureDef.density = 10000;
			floorFixtureDef.friction = 1;

			m_body = world.CreateBody(&floorDef);
			m_body->CreateFixture(&floorFixtureDef);
		}

	public:
		b2Body* m_body;
		b2Vec2 m_size;
		b2Vec2 m_position;
	};
}