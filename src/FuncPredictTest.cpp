//
// Created by negi on 08.01.17.
//

#include <iostream>
#include <SFML/Graphics/Text.hpp>
#include "FuncPredictTest.h"

FuncPredictTest::FuncPredictTest() :
        model(0.3, 1.8), trainSet(DOTS) {
    if (!font.loadFromFile("res/Roboto-Regular.ttf")) {
        exit(EXIT_FAILURE);
    }
    model.addLayer(new Layer(1,34));
    model.addLayer(new RNNLayer(34,8, trainSamplesCount));
    model.addLayer(new RNNLayer(8,1, trainSamplesCount));
    model.addLayer(new Layer(1,1));

    trainSetPredicted.reserve(DOTS);

    for (int i = 0; i < DOTS; ++i) {
        double x = SCREEN_WIDTH * 2.0 / (DOTS - 1) * i;
        double y = function(x);
        trainSet.add(x, y);
    }
    trainSet.normalize();
}

void FuncPredictTest::train() {
}

void FuncPredictTest::update() {
    for (int i = 0; i < 10; i++) {
        std::vector<matrix> XSet;
        XSet.reserve((unsigned long)trainSamplesCount);
        std::vector<matrix> YSet;
        YSet.reserve((unsigned long)trainSamplesCount);

        int r = Utils::randInt(0, DOTS - trainSamplesCount);

        for (int pos = r; pos < r + trainSamplesCount; pos++) {
            matrix X = matrix(1,1);
            matrix Y = matrix(1,1);
            X[0][0] = trainSet.setX[pos];
            Y[0][0] = trainSet.setY[pos];
            XSet.push_back(X);
            YSet.push_back(Y);
        }

        model.train(XSet, YSet, predicted);

        for (int j = 0; j < trainSamplesCount; j++) {
            trainSetPredicted[r + j] += (predicted[j] - trainSetPredicted[r + j]) * 0.1;
        }
    }
}

void FuncPredictTest::draw(sf::RenderWindow& window) {
    if (DOTS < 3) {
        return;
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

    for (unsigned int i = 0; i < DOTS; ++i) {
        if (num > 1) {
            num++;
            linePredicted[num - 1] = prev;
        }

        double x = trainSet.unpackX(i); //trainSetX[i] * (setXMax - setXMin) + setXMin;
        double y = trainSet.unpack(trainSetPredicted[i]); //trainSetPredicted[i] * (setYMax - setYMin) + setYMin;
        sf::Vertex vert = sf::Vertex(sf::Vector2f((float)x, (float)y));
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
