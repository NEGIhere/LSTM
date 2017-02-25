//
// Created by negi on 08.01.17.
//

#include <iostream>
#include "FuncPredictTest.h"
#include "utils/Config.h"
#include "../old/Net.h"

FuncPredictTest::FuncPredictTest() {

}

void FuncPredictTest::update() {

}

void FuncPredictTest::draw(sf::RenderWindow& window) {
    if (DOTS < 3) {
        return;
    }

    unsigned int vertexCount = (DOTS - 1) * 2;

    sf::Vertex line[vertexCount];
    sf::Vertex prev;
    unsigned int num = 0;

    for (int i = 0; i < DOTS; ++i) {
        float x = SCREEN_WIDTH * 2 / (DOTS - 1) * i;

        if (num > 1) {
            num++;
            line[num - 1] = prev;
        }

        sf::Vertex vert = sf::Vertex(sf::Vector2f(x, (float)function(x)));
        vert.color = sf::Color::Black;

        line[num] = vert;
        prev = vert;
        num++;
    }
    window.draw(line, DOTS, sf::Lines);
}
