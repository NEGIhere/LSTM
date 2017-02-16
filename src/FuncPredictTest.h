//
// Created by negi on 08.01.17.
//

#ifndef LSTM_FUNCPREDICTTEST_H
#define LSTM_FUNCPREDICTTEST_H


#include <SFML/Graphics/RenderWindow.hpp>
#include <cmath>
#include "utils/Config.h"

class FuncPredictTest {
public:
    FuncPredictTest();
    static const unsigned int DOTS = 35;
    void update();
    void draw(sf::RenderWindow& window);

    static double inline function(double x) {
        return cos(x / SCREEN_WIDTH * 10) * SCREEN_HEIGHT / 3 + SCREEN_HEIGHT / 2;
    }
};


#endif //LSTM_FUNCPREDICTTEST_H
