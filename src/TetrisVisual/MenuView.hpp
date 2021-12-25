#pragma once

#include <SFML/Graphics.hpp>

namespace pt
{
	class MenuView
	{
	public:
		MenuView(sf::RenderWindow& window) :
			m_window(window),
			m_background(),
			m_font(),
			m_text(),
			m_time(0)
		{
			m_background = Resources::getInstance().BackgroundTex;
			m_background.setRepeated(true);

			m_font = Resources::getInstance().Font;
			
			textPrepaire();
		}

		void update(float dt)
		{
			float k = m_time += 1.f / 100.f;
			m_text.setScale(1.f + std::sinf(k) / 20.f, 1.f + std::sinf(k) / 10.f);
		}

		void draw()
		{
			m_window.draw(sf::Sprite(m_background));
			m_window.draw(m_text);
		}

	private:
		sf::RenderWindow& m_window;
		sf::Texture m_background;
		sf::Font m_font;
		sf::Text m_text;

		float m_time;

		void textPrepaire()
		{
			m_text.setString(L"\t  Типо меню .\nPress any key to start.");
			m_text.setFont(m_font);
			m_text.setCharacterSize(50u);
			m_text.setOutlineThickness(3.f);
			m_text.setOutlineColor(sf::Color::Black);
			m_text.setFillColor(sf::Color::White);

			sf::FloatRect textRect = m_text.getLocalBounds();
			m_text.setOrigin(textRect.left + textRect.width / 2.f,
				textRect.top + textRect.height / 2.0f);
			m_text.setPosition(sf::Vector2f(m_window.getSize().x / 2.f, m_window.getSize().y / 2.f));
		}
	};
}