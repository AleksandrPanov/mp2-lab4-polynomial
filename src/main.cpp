#ifndef _DEBUG
#define SFML_STATIC
#pragma comment(linker, "/SUBSYSTEM:WINDOWS")
#endif

#include <iostream>

#include <SFML/Graphics.hpp>
#include "../extern/box2d/include/box2d/box2d.h"

#include "TetrisLogic/Tetris.hpp"
#include "TetrisVisual/TetrisView.hpp"
#include "TetrisVisual/MenuView.hpp"
#include "Scene.hpp"

#include "States/MenuState.hpp"
#include "TetrisLogic/Rules/StandartRules.hpp"

#include "Resources.hpp"



int main()
{
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8u;

	sf::RenderWindow window(sf::VideoMode(600, 800), "Physical Tetris | v1.1 betha", sf::Style::Close, settings);
	pt::Tetris tetris(3.f);
	pt::TetrisView view(window, tetris, 10.f);
	pt::MenuView menu(window);
	pt::Scene scene(tetris, view, menu);

	window.setKeyRepeatEnabled(true);
	window.setFramerateLimit(60);

	tetris.setRules(new pt::StandartRules(&tetris));
	scene.setState(new pt::MenuState(&scene));

	srand(time(0));
	sf::Clock worldClock;
	sf::Clock spawnClock;
	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			scene.checkEvent(event);
		}

		scene.update(worldClock.restart().asSeconds());
		
		if (spawnClock.getElapsedTime().asSeconds() > 5.f)
		{
			spawnClock.restart();
		}

		window.clear();
		scene.draw();
		window.display();
	}

	return 0;
}