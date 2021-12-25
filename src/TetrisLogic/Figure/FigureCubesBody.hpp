#pragma once

#include <iostream>
#include <cmath>

#include "../../../extern/box2d/include/box2d/box2d.h"

#include "Figures.hpp"
#include "IFigure.hpp"

namespace pt
{
	class FigureCubesBody : public IFigureBody
	{
	public:
		FigureArray m_array;
		float m_cubeSize;
		std::vector<b2Body*> m_bodies;

		FigureCubesBody(b2World& world, const FigureArray array, float cubeSize) :
			m_world(world),
			m_array(array),
			m_cubeSize(cubeSize)
		{
			createCubesFigureByArray();
		}

		FigureCubesBody(b2World& world, float cubeSize) :
			FigureCubesBody(world, getRandomArray(), cubeSize)
		{	}

		void move(b2Vec2 offset) override
		{
			b2Vec2 pos = m_bodies[0]->GetPosition();
			setPosition(pos + offset);
		}

		void rotate(float angle) override
		{
			b2Vec2 pos = m_bodies[0]->GetPosition();
			float newAngle = m_bodies[0]->GetAngle() + angle;
			for (int i = 0; i < m_bodies.size(); ++i)
			{
				auto& body = m_bodies[i];
				auto& offset = m_offsets[i];
				b2Vec2 newOff;
				newOff.x = offset.x * std::cosf(newAngle) - offset.y * std::sinf(newAngle);
				newOff.y = offset.x * std::sinf(newAngle) + offset.y * std::cosf(newAngle);
				body->SetTransform(pos + newOff, newAngle);
			}
		}

		void setPosition(b2Vec2 pos) override
		{
			for (int i = 0; i < m_bodies.size(); ++i)
			{
				auto& body = m_bodies[i];
				auto& offset = m_offsets[i];
				float angle = m_bodies[i]->GetAngle();
				b2Vec2 newOff;
				newOff.x = (offset.x * std::cos(angle) - offset.y * std::sin(angle));
				newOff.y = (offset.x * std::sin(angle) + offset.y * std::cos(angle));
				body->SetTransform(pos + newOff, angle);
			}
			//std::cout << m_bodies[0]->GetPosition().x << " " << m_bodies[1]->GetPosition().x << " " << m_bodies[2]->GetPosition().x << " " << m_bodies[3]->GetPosition().x << " " << std::endl;
		}

		void setAngularVelocity(float velocity)
		{
			for (auto& body : m_bodies)
				body->SetAngularVelocity(velocity);
		}

		void setVelocity(b2Vec2 velocity)
		{
			for (auto& body : m_bodies)
				body->SetLinearVelocity(velocity);
		}

		void setGravityScale(float scale)
		{
			for (auto& body : m_bodies)
				body->SetGravityScale(scale);
		}

		float getAngle()
		{
			return m_bodies[0]->GetAngle();
		}

		b2Vec2 getPosition()
		{
			b2Vec2 newOff; float angle = m_bodies[0]->GetAngle();
			newOff.x = (m_offsets[0].x * std::cos(angle) - m_offsets[0].y * std::sin(angle));
			newOff.y = (m_offsets[0].x * std::sin(angle) + m_offsets[0].y * std::cos(angle));
			return m_bodies[0]->GetPosition() - newOff;
		}


	private:
		b2World& m_world;
		std::vector<b2Vec2> m_offsets;

		b2Body* createCube()
		{
			b2Body* cube;
			b2BodyDef cubeDef;
			b2Fixture* cubeFixture;
			b2FixtureDef cubeFixtureDef;
			b2PolygonShape polygon;

			cubeDef.type = b2BodyType::b2_dynamicBody;
			cube = m_world.CreateBody(&cubeDef);

			polygon.SetAsBox(m_cubeSize * 0.98f / 2, m_cubeSize * 0.98f / 2);
			cubeFixtureDef.restitutionThreshold = 0;
			cubeFixtureDef.restitution = 0;
			cubeFixtureDef.friction = 1;
			cubeFixtureDef.density = 10000.f;
			cubeFixtureDef.shape = &polygon;
			cubeFixture = cube->CreateFixture(&cubeFixtureDef);

			return cube;
		}

		void createCubesFigureByArray()
		{
			b2Body* currentCube = createCube();
			b2Body* previousCube = nullptr;

			int iPrev, jPrev;
			int iFirst, jFirst;
			for (int iRow = 0; iRow < m_array.size(); ++iRow)
			{
				for (int jColumn = 0; jColumn < m_array[0].size(); ++jColumn)
				{
					if (m_array[iRow][jColumn])
					{
						if (previousCube == nullptr)
						{
							previousCube = currentCube;
							iFirst = m_array.size() / 2;
							jFirst = m_array[0].size() / 2;
						}
						else
						{
							previousCube = currentCube;
							currentCube = createCube();

							b2WeldJointDef jointDef;
							jointDef.collideConnected = false;
							jointDef.bodyA = previousCube;
							jointDef.bodyB = currentCube;
							jointDef.localAnchorB.Set(-(jColumn - jPrev) * m_cubeSize, -(iRow - iPrev) * m_cubeSize);
							m_world.CreateJoint(&jointDef);
						}
						iPrev = iRow;
						jPrev = jColumn;
						m_offsets.push_back(b2Vec2
						((jColumn - jFirst) * m_cubeSize,
							(iRow - iFirst) * m_cubeSize)
						);
						m_bodies.push_back(currentCube);
					}
				}
			}

			/*currentCube = nullptr;
			previousCube = nullptr;
			int count = 0;
			for (int jColumn = m_array[0].size() - 1; jColumn >= 0; --jColumn)
			{
				for (int iRow = m_array.size() - 1; iRow >= 0; --iRow)
				{
					if (m_array[iRow][jColumn])
					{
						if (currentCube == nullptr)
						{
							currentCube = m_bodies[m_bodies.size() - 1 - count++];
						}
						else
						{
							previousCube = currentCube;
							currentCube = m_bodies[m_bodies.size() - 1 - count++];

							b2WeldJointDef jointDef;
							jointDef.collideConnected = false;
							jointDef.bodyA = previousCube;
							jointDef.bodyB = currentCube;
							jointDef.localAnchorB.Set(-(jColumn - jPrev) * m_cubeSize, -(iRow - iPrev) * m_cubeSize);
							m_world.CreateJoint(&jointDef);
						}
						iPrev = iRow;
						jPrev = jColumn;
					}
				}
			}*/
		}

		/*
		enum EDirection
		{
			S, D, W, A
		};

		enum EEdgeDirection
		{
			WA, AS, SD, DW
		};

		struct Direction
		{
			Direction() { }

			Direction(EDirection dir)
			{
				switch (dir)
				{
				case pt::FigureCubesBody::S: x = 0; y = 1; break;
				case pt::FigureCubesBody::D: x = 1; y = 0; break;
				case pt::FigureCubesBody::W: x = 0; y = -1; break;
				case pt::FigureCubesBody::A: x = -1; y = 0; break;
				}
			}

			Direction(EEdgeDirection dir)
			{
				switch (dir)
				{
				case pt::FigureCubesBody::WA: x = -1; y = -1; break;
				case pt::FigureCubesBody::AS: x = -1; y = 1; break;
				case pt::FigureCubesBody::SD: x = 1; y = 1; break;
				case pt::FigureCubesBody::DW: x = 1; y = -1; break;
				}
			}

			int x, y;
		};

		void createShapeFigureByArray()
		{
			int xFirst, yFirst;

			for (int y = 0; y < m_array.size(); ++y)
			{
				for (int x = 0; x < m_array[0].size(); ++x)
				{
					if (m_array[y][x])
					{
						xFirst = x;
						yFirst = y;
						goto afterFirst;
					}
				}
			}

		afterFirst:
			arrayBypass(xFirst, yFirst, xFirst, yFirst, EDirection(0), EEdgeDirection(1), true);

			b2PolygonShape polygon;
			polygon.Set(&figurePoints[0], figurePoints.size());

			b2BodyDef shapeDef;
			b2Fixture* shapeFixture;
			b2FixtureDef shapeFixtureDef;

			shapeDef.type = b2BodyType::b2_dynamicBody;
			shape = m_world.CreateBody(&shapeDef);

			shapeFixtureDef.density = 1;
			shapeFixtureDef.shape = &polygon;
			shapeFixture = shape->CreateFixture(&shapeFixtureDef);
		}

		void arrayBypass(int curX, int curY, int xFirst, int yFirst, EDirection dir, EEdgeDirection eDir, bool first = false)
		{
			if (curX == xFirst && curY == yFirst && !first && dir == S)
				return;

			int dirOffset = -2;
			int edgeOffset = -2;
			Direction moveDir;
			EDirection nextDir;
			int offX, offY;
			do
			{
				++dirOffset;
				++edgeOffset;
				nextDir = EDirection((dir + dirOffset + 4) % 4);
				moveDir = nextDir;
				offX = moveDir.x;
				offY = moveDir.y;

				if (curX == xFirst && curY == yFirst && !first && nextDir == S)
					return;

				if (needEdgeByDirection(curX, curY, EEdgeDirection(eDir + edgeOffset)))
				{
					b2Vec2 point;
					switch (EEdgeDirection(eDir + edgeOffset))
					{
					case pt::FigureCubesBody::WA: point.x = curX - xFirst; point.y = curY - yFirst; break;
					case pt::FigureCubesBody::AS: point.x = curX - xFirst; point.y = curY - yFirst + 1; break;
					case pt::FigureCubesBody::SD: point.x = curX - xFirst + 1; point.y = curY - yFirst + 1; break;
					case pt::FigureCubesBody::DW: point.x = curX - xFirst + 1; point.y = curY - yFirst; break;
					}
					point *= m_cubeSize;
					uniquePushBack(figurePoints, point);
				}
			} while (!hasCubeByOffset(curX, curY, offX, offY));

			arrayBypass(curX + offX, curY + offY, xFirst, yFirst, nextDir, EEdgeDirection(1 + (int)nextDir));
		}

		bool needEdgeByDirection(int curX, int curY, EEdgeDirection dir)
		{
			int offX = 0, offY = 0;
			switch (dir)
			{
			case pt::FigureCubesBody::WA: offX = -1; offY = -1; break;
			case pt::FigureCubesBody::AS: offX = -1; offY = 1; break;
			case pt::FigureCubesBody::SD: offX = 1; offY = 1; break;
			case pt::FigureCubesBody::DW: offX = 1; offY = -1; break;
			}

			if (hasCubeByOffset(curX, curY, 0, offY) &&
				hasCubeByOffset(curX, curY, offX, 0) &&
				hasCubeByOffset(curX, curY, offX, offY)
				)
				return false;

			if (hasCubeByOffset(curX, curY, 0, offY) &&
				hasCubeByOffset(curX, curY, offX, 0))
				return true;

			if (hasCubeByOffset(curX, curY, offX, offY))
				return true;

			if (!hasCubeByOffset(curX, curY, 0, offY) &&
				!hasCubeByOffset(curX, curY, offX, 0))
				return true;

			return false;
		}

		bool hasCubeByOffset(int curX, int curY, int offX, int offY)
		{
			int x = curX + offX;
			int y = curY + offY;

			if (x >= m_array[0].size() || x < 0 ||
				y >= m_array.size()    || y < 0)
				return false;

			return m_array[y][x];
		}

		void uniquePushBack(std::vector<b2Vec2>& vector, b2Vec2& point)
		{
			auto it = std::find(vector.begin(), vector.end(), point);
			if (it == vector.end()) vector.push_back(point);
		}
		*/
	};
}