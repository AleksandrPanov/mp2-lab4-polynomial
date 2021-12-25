#pragma once

#include "../../../extern/box2d/include/box2d/box2d.h"

#include "Figures.hpp"
#include "IFigure.hpp"

namespace pt
{
	class FigureShapeBody : public IFigureBody
	{
	public:
		b2Body* m_body;
		FigureArray m_array;
		float m_cubeSize;

		FigureShapeBody(b2World& world, const FigureArray figureArray, float cubeSize) :
			m_world(world),
			m_array(figureArray),
			m_cubeSize(cubeSize),
			m_id(objectCount++)
		{
			createBody();
		}

		FigureShapeBody(b2World& world, float cubeSize) :
			FigureShapeBody(world, getRandomArray(), cubeSize)
		{	}

		~FigureShapeBody()
		{
			m_world.DestroyBody(m_body);
		}

		bool isCollide() override
		{
			for (auto edge = m_body->GetContactList(); edge; edge = edge->next)
				if (edge->contact->IsTouching())
					return true;
			return false;
		}

		void rotate(float angle) override
		{
			auto& pos = m_body->GetPosition();
			auto newAngle = m_body->GetAngle() + angle;
			m_body->SetTransform(pos, newAngle);
		}

		void setPosition(b2Vec2 pos) override
		{
			m_body->SetTransform(pos, m_body->GetAngle());
		}

		void setAngularVelocity(float velocity) override
		{
			m_body->SetAngularVelocity(velocity);
		}

		void setVelocity(b2Vec2 velocity) override
		{
			m_body->SetLinearVelocity(velocity);
		}

		void setGravityScale(float scale) override
		{
			m_body->SetGravityScale(scale);
		}

		b2Vec2 getPosition() override
		{
			return m_body->GetPosition();
		}

		float getAngle() override
		{
			return m_body->GetAngle();
		}
		
		int getId() override
		{
			return m_id;
		}

		b2AABB getBounds() override
		{
			b2AABB aabb;
			b2Transform transform;
			transform = m_body->GetTransform();
			aabb.lowerBound = b2Vec2(FLT_MAX, FLT_MAX);
			aabb.upperBound = b2Vec2(-FLT_MAX, -FLT_MAX);
			for (b2Fixture* fixture = m_body->GetFixtureList(); fixture; fixture = fixture->GetNext())
			{
				const b2Shape* shape = fixture->GetShape();
				const int childCount = shape->GetChildCount();
				for (int child = 0; child < childCount; ++child) {
					const b2Vec2 r(shape->m_radius, shape->m_radius);
					b2AABB shapeAABB;
					shape->ComputeAABB(&shapeAABB, transform, child);
					shapeAABB.lowerBound = shapeAABB.lowerBound + r;
					shapeAABB.upperBound = shapeAABB.upperBound - r;
					aabb.Combine(shapeAABB);
				}
			}
			return aabb;
		}

	private:
		b2World& m_world;
		int m_id;
		static int objectCount;

		void createBody()
		{
			b2BodyDef def; def.type = b2BodyType::b2_dynamicBody;
			m_body = m_world.CreateBody(&def);

			b2FixtureDef fixDef;
			fixDef.density = 1.f;
			fixDef.friction = 1.f;
			fixDef.restitution = 0.f;
			fixDef.restitutionThreshold = 0.f;

			bool isFirst = true;
			int iPrev, jPrev;
			float iOrigin, jOrigin;
			for (int iRow = 0; iRow < m_array.size(); ++iRow)
			{
				for (int jColumn = 0; jColumn < m_array[0].size(); ++jColumn)
				{
					if (m_array[iRow][jColumn])
					{
						if (isFirst)
						{
							isFirst = false;
							iOrigin = (m_array.size() - 1) / 2.f; // origin
							jOrigin = (m_array[0].size() - 1) / 2.f;
						}

						b2PolygonShape cubeShape;
						float size = (m_cubeSize / 2.f) * 0.97f;
						cubeShape.SetAsBox(size, size);
						for (int i = 0; i < cubeShape.m_count; ++i)
						{
							auto& edge = cubeShape.m_vertices[i];
							edge += b2Vec2((jColumn - jOrigin) * m_cubeSize, (iRow - iOrigin) * m_cubeSize);
						}

						fixDef.shape = &cubeShape;
						m_body->CreateFixture(&fixDef);
					}
				}
			}
		}
	};

	int FigureShapeBody::objectCount = 0;
}