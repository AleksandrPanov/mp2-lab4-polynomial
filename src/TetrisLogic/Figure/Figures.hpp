#pragma once

#include <vector>
#include <array>
#include "../../Vector2f.hpp"

namespace pt
{
	const float PI = 3.1415926535f;

	enum class FigureType
	{
		LFormShape,
		QudricFormShape,
		ZFormShape,
		TFormShape
	};

	typedef std::vector<std::vector<bool>> FigureArray;
	const std::vector<FigureArray> figureArrays = {
		{
			{0,0,0,0},
			{0,1,1,0},
			{0,0,1,0},
			{0,0,1,0},
			{0,0,0,0}
		},
		
		{
			{0,0,0,0},
			{0,1,1,0},
			{0,1,0,0},
			{0,1,0,0},
			{0,0,0,0}
		},

		{
			{0,0,0,0},
			{0,1,0,0},
			{0,1,1,0},
			{0,0,1,0},
			{0,0,0,0}
		},

		{
			{0,0,0,0,0},
			{0,1,1,0,0},
			{0,0,1,1,0},
			{0,0,0,0,0},
		},

		{
			{0,0,1,0,0},
			{0,0,1,0,0},
			{0,0,1,0,0},
			{0,0,1,0,0}
		},

		{
			{0,0,1,0,0},
			{0,0,1,0,0},
			{0,0,1,0,0},
			{0,0,1,0,0},
			{0,0,1,0,0}
		},

		{
			{0,0,0,0},
			{0,1,1,0},
			{0,1,1,0},
			{0,0,0,0}
		},

		{
			{0,0,0,0,0},
			{0,1,1,1,0},
			{0,1,1,1,0},
			{0,1,1,1,0},
			{0,0,0,0,0}
		},

		{
			{0,0,0,0,0},
			{0,0,1,0,0},
			{0,1,1,1,0},
			{0,0,0,0,0}
		}
		
	};

	typedef std::vector<b2Vec2> FigurePoint;
	const std::vector<FigurePoint> figurePoints = {
		{	// Z
			{0.f, 0.f},
			{1.f, 0.f},
			{1.f, 1.f},
			{2.f, 1.f},
			{2.f, 3.f},
			{1.f, 3.f},
			{1.f, 2.f},
			{0.f, 2.f},
		},

		{	// J
			{0.f, 0.f},
			{3.f, 0.f},
			{3.f, 1.f},
			{1.f, 1.f},
			{1.f, 2.f},
			{0.f, 2.f},
		},

		{	// T
			{1.f, 0.f},
			{2.f, 0.f},
			{2.f, 1.f},
			{3.f, 1.f},
			{3.f, 2.f},
			{0.f, 2.f},
			{0.f, 1.f},
			{1.f, 1.f},
		},

		{	// I
			{0.f, 0.f},
			{0.f, 4.f},
			{1.f, 4.f},
			{1.f, 0.f},
		},

		{	// square
			{0.f, 0.f},
			{0.f, 2.f},
			{2.f, 2.f},
			{2.f, 0.f},
		},
	};

	FigureArray getRandomArray()
	{
		auto type = rand() % figureArrays.size();
		return figureArrays[type];
	}

	FigurePoint getRandomPoint()
	{
		auto type = 1;// rand() % figurePoints.size();
		return figurePoints[type];
	}
}