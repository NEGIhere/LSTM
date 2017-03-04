//
// Created by negi on 08.01.17.
//

#ifndef LSTM_FUNCPREDICTTEST_H
#define LSTM_FUNCPREDICTTEST_H


#include <SFML/Graphics/RenderWindow.hpp>
#include <cmath>
#include "utils/Config.h"
#include "Model.h"
#include "utils/DataSet.h"

class FuncPredictTest {
public:
    FuncPredictTest();
    static const unsigned int DOTS = 35 * 2;
    void update();
    void draw(sf::RenderWindow& window);
    void train();
    static double inline function(double x) {
        return cos(x / SCREEN_WIDTH * 10/* * (x / SCREEN_WIDTH * 2)*/) /* * (x / SCREEN_WIDTH)*/ * SCREEN_HEIGHT / 3 + SCREEN_HEIGHT / 2;
    }
private:
    static const int trainSamplesCount = 8;
    double predicted[trainSamplesCount];
    std::vector<double> /*trainSetX, trainSetY,*/ trainSetPredicted;
    //double setXMin, setXMax, setYMin, setYMax;
    DataSet trainSet;
    sf::Font font;
    sf::Text text;
    Model model;
};


#endif //LSTM_FUNCPREDICTTEST_H
