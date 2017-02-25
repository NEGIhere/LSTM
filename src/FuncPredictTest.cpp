//
// Created by negi on 08.01.17.
//

#include <iostream>
#include <SFML/Graphics/Text.hpp>
#include "FuncPredictTest.h"
#include "utils/Config.h"
#include "Model.h"

const int trainSamplesCount = 10;
double d[trainSamplesCount];
std::vector<double> trainSetX, trainSetY, trainSetPredicted;
double setXMin, setXMax, setYMin, setYMax;
Model model = Model(0.3, 1.8);
sf::Font font;

FuncPredictTest::FuncPredictTest() {
    if (!font.loadFromFile("res/Roboto-Regular.ttf")) {
        exit(1);
    }
    model.addLayer(new Layer(1,34));
    model.addLayer(new RNNLayer(34,8, trainSamplesCount));
    model.addLayer(new RNNLayer(8,1, trainSamplesCount));
    model.addLayer(new Layer(1,1));

    setXMin = setYMin = 10000000.0;
    setXMax = setYMax = -10000000.0;

    trainSetPredicted.reserve(DOTS);

    for (int i = 0; i < DOTS; ++i) {
        double x = SCREEN_WIDTH * 2.0 / (DOTS - 1) * i;
        double y = function(x);
        trainSetX.push_back(x);
        trainSetY.push_back(y);

        if (setXMin > x) {
            setXMin = x;
        } else if (setXMax < x) {
            setXMax = x;
        }

        if (setYMin > y) {
            setYMin = y;
        } else if (setYMax < y) {
            setYMax = y;
        }
    }

    for (int i = 0; i < trainSetX.size(); i++) {
        trainSetX[i] = (trainSetX[i] - setXMin) / (setXMax - setXMin);
        trainSetY[i] = (trainSetY[i] - setYMin) / (setYMax - setYMin);
    }
}

void FuncPredictTest::train() {

}

void FuncPredictTest::update() {

}

void FuncPredictTest::draw(sf::RenderWindow& window) {
    if (DOTS < 3) {
        return;
    }
    for (int i = 0; i < 10; i++) {
        std::vector<matrix> XSet;
        XSet.reserve((unsigned long)trainSamplesCount);
        std::vector<matrix> YSet;
        YSet.reserve((unsigned long)trainSamplesCount);

        int r = Utils::randInt(0, (int)trainSetX.size() - trainSamplesCount);

        for (int pos = r; pos < r + trainSamplesCount; pos++) {
            matrix X = matrix(1,1);
            matrix Y = matrix(1,1);
            X[0][0] = trainSetX[pos];
            Y[0][0] = trainSetY[pos];
            XSet.push_back(X);
            YSet.push_back(Y);
        }

        model.train(XSet, YSet, d, (unsigned int)trainSamplesCount);

        if (i % 1 == 0) {
            //Utils::print(std::string("Error:") + std::to_string(model.getRecentAverageError()));
            //Utils::print(std::string("Pred:") + std::to_string(d[0]) + ", " + std::to_string(d[1]) + ", " + std::to_string(d[2]));
            //Utils::print(std::string("True:") + std::to_string(YSet[0][0][0]) + ", " + std::to_string(YSet[1][0][0]) + ", " + std::to_string(YSet[2][0][0]));
            for (int j = 0; j < trainSamplesCount; j++) {
                trainSetPredicted[r + j] += (d[j] - trainSetPredicted[r + j]) * 0.1;
            }
            //Utils::print(std::to_string(a_int) + " + " + std::to_string(b_int) + " = " + std::to_string(out));
        }
    }

    unsigned int vertexCount = (DOTS - 1) * 2;

    sf::Vertex line[vertexCount];
    sf::Vertex linePredicted[vertexCount];
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

    prev.position.x = prev.position.y = 0;
    num = 0;

    for (int i = 0; i < DOTS; ++i) {
        //float x = SCREEN_WIDTH * 2 / (DOTS - 1) * i;

        if (num > 1) {
            num++;
            linePredicted[num - 1] = prev;
        }

        double xx = trainSetX[i] * (setXMax - setXMin) + setXMin;
        double yy = trainSetPredicted[i] * (setYMax - setYMin) + setYMin;
        sf::Vertex vert = sf::Vertex(sf::Vector2f((float)xx, (float)yy));
        vert.color = sf::Color::Red;

        linePredicted[num] = vert;
        prev = vert;
        num++;
    }
    window.draw(line, DOTS, sf::Lines);
    window.draw(linePredicted, DOTS, sf::Lines);
    sf::Text text("Loss:" + sf::String(std::to_string(model.getRecentAverageError())), font);
    text.setPosition(10, 10);
    text.setColor(sf::Color::Black);
    window.draw(text);
}
