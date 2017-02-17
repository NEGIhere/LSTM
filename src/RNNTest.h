//
// Created by negi on 11.02.17.
//

#ifndef LSTM_RNNTEST_H
#define LSTM_RNNTEST_H


#include <SFML/Graphics/RenderWindow.hpp>
#include "../old/Net.h"

class RNNTest {
public:

    RNNTest();

    void update();
    void draw(sf::RenderWindow& window);

private:
    Net* net;
};


#endif //LSTM_RNNTEST_H
