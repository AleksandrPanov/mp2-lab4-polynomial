#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include <Windows.h>

#include "../resource.h"

namespace pt
{
	class Resources
	{
	public:
		sf::Texture BackgroundTex;
		sf::Texture FloorTex;
		sf::Texture FigureTex;

		sf::Font Font;

		sf::Sound FigureSpawnSound;
		sf::Sound FigureCollishionSound;

		sf::Music BackMusic;

		static Resources& getInstance()
		{
			static Resources resource;
			return resource;
		}

	private:
		sf::SoundBuffer m_figureSpawnSoundBuffer;
		sf::SoundBuffer m_FigureCollishionSoundBuffer;

		Resources()
		{
			HRSRC find;
			HGLOBAL res;
			DWORD size;

			find = FindResource(NULL, MAKEINTRESOURCE(IDB_PNG1), "PNG");
			size = SizeofResource(NULL, find);
			res = LoadResource(NULL, find);
			BackgroundTex.loadFromMemory(res, size);

			find = FindResource(NULL, MAKEINTRESOURCE(IDB_PNG2), "PNG");
			size = SizeofResource(NULL, find);
			res = LoadResource(NULL, find);
			FloorTex.loadFromMemory(res, size);

			find = FindResource(NULL, MAKEINTRESOURCE(IDB_PNG3), "PNG");
			size = SizeofResource(NULL, find);
			res = LoadResource(NULL, find);
			FigureTex.loadFromMemory(res, size);

			find = FindResource(NULL, MAKEINTRESOURCE(IDR_MY_FONT2), "MY_FONT");
			res = LoadResource(NULL, find);
			size = SizeofResource(NULL, find);
			Font.loadFromMemory(res, size);

			find = FindResource(NULL, MAKEINTRESOURCE(IDR_MY_OGG2), "MY_OGG");
			res = LoadResource(NULL, find);
			size = SizeofResource(NULL, find);
			m_figureSpawnSoundBuffer.loadFromMemory(res, size);
			FigureSpawnSound.setBuffer(m_figureSpawnSoundBuffer);

			find = FindResource(NULL, MAKEINTRESOURCE(IDR_MY_OGG1), "MY_OGG");
			res = LoadResource(NULL, find);
			size = SizeofResource(NULL, find);
			m_FigureCollishionSoundBuffer.loadFromMemory(res, size);
			FigureCollishionSound.setBuffer(m_FigureCollishionSoundBuffer);

			find = FindResource(NULL, MAKEINTRESOURCE(IDR_MY_OGG3), "MY_OGG");
			res = LoadResource(NULL, find);
			size = SizeofResource(NULL, find);
			BackMusic.openFromMemory(res, size);
		}
	};
}