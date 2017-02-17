#include <SFML/Graphics.hpp>
#include <iostream>
#include "utils/Config.h"
#include "FuncPredictTest.h"
#include "utils/Utils.h"
#include "RNNTest.h"
#include "utils/matrix.h"

sf::RenderWindow* window;

FuncPredictTest* test;

void update() {
    test->update();

}

void draw() {
    test->draw(*window);
}

void sghhit(const unsigned int a, const unsigned int b) {
    std::cout << a << b;
}

int main() {
    int a = 2;

    matrix mat = 2 * matrix::random::rand(4,4) - 1.0;
    //Utils::print(mat);

    srand(1); //
    RNNTest* rnn = new RNNTest();

    bool rendering = true;
    test = new FuncPredictTest();
    sf::ContextSettings settings;
    settings.antialiasingLevel = 0;

    window = new sf::RenderWindow(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "LSTM", sf::Style::Default, settings);
    window->setVerticalSyncEnabled(false);
    window->setPosition(sf::Vector2i(1100, 100));
    window->setFramerateLimit(60);

    sf::View view = window->getDefaultView();
    view.zoom(1.0f);
    window->setView(view);

    while (window->isOpen()) {
        sf::Event event;

        while (window->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window->close();
                break;
            } else if (event.type == sf::Event::MouseButtonPressed) {
                int mouseX = sf::Mouse::getPosition(*window).x;
                int mouseY = sf::Mouse::getPosition(*window).y;

                if (event.mouseButton.button == sf::Mouse::Left) {
                    rendering = !rendering;
                }
            } else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::S) {
                    //balance->save();
                } else if (event.key.code == sf::Keyboard::L) {
                    //balance->load();
                }
            }
        }
        update();
        window->clear(sf::Color::White);
        if (rendering) {
            draw();
        }
        window->display();
    }

    delete window;
    delete test;
    /*std::vector<long double> arr(10, MAXDOUBLE);
    for (int i = 0; i < 40; i++) {
        for (long double num : arr) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        arr.push_back(rand() % 10);
        if (arr.size() > 10) {
            arr.erase(arr.begin());
        }
    }*/

    return 0;
}