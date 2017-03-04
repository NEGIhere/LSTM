//
// Created by negi on 08.01.17.
//

#include <iostream>
#include <SFML/Graphics/Text.hpp>
#include <chrono>
#include "FuncPredictTest.h"

std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
unsigned long iters = 0;

FuncPredictTest::FuncPredictTest() :
        model(0.01, 0.8, 0.0), trainSet(DOTS) {
    if (!font.loadFromFile("res/Consolas.ttf")) {
        exit(EXIT_FAILURE);
    }
    model.addLayer(new Layer(1,31));
    model.addLayer(new RNNLayer(31,8, trainSamplesCount));
    model.addLayer(new RNNLayer(8,1, trainSamplesCount));
    model.addLayer(new Layer(1,1));

    trainSetPredicted.reserve(DOTS);

    for (int i = 0; i < DOTS; ++i) {
        double x = SCREEN_WIDTH * 2.0 / (DOTS - 1) * i;
        double y = function(x);
        trainSet.add(x, y);
    }
    trainSet.normalize();
    text = sf::Text("", font);
    text.setPosition(8, 1);
    text.setScale(sf::Vector2f(0.7f,0.7f));
    text.setColor(sf::Color::Black);

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
        iters++;

        for (int j = 0; j < trainSamplesCount; j++) {
            trainSetPredicted[r + j] += (predicted[j] - trainSetPredicted[r + j]) * 0.1;
        }
    }
}


void FuncPredictTest::draw(sf::RenderWindow& window) {
    static_assert(DOTS > 3,  "DOTS must be >3");

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
    long elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t0).count();
    text.setString("Loss:" + sf::String(std::to_string(model.getRecentAverageError())) + ", eta:" + sf::String(std::to_string(model.eta)) + ", mu:" + sf::String(std::to_string(1.0 - model.mu)) + "\nSecs:" + sf::String(std::to_string(elapsedSeconds)) + ", Iters:" + sf::String(std::to_string(iters)));
    window.draw(text);
}
