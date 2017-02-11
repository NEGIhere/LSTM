//
// Created by negi on 11.02.17.
//

#include <iostream>
#include <cstring>
#include "RNNTest.h"
#include "Utils.h"

RNNTest::RNNTest() {
    net = new Net({2, 16, 1});
    std::vector<double> weights;
    net->getWeights(weights, false);
    //Utils::print(weights);
    printf("%.16f\n", Neuron::sigmoid(1.51251924189248));

    int binaryDim = 8;
    int largestNum = (int)pow(2, binaryDim);
    int* int2bin[largestNum];

    for (int i = 0; i < largestNum; i++) {
        int2bin[i] = new int[binaryDim];

        int num = i;
        for (int j = binaryDim - 1; j >= 0; --j) {
            int2bin[i][j] = num % 2;
            num /= 2;
        }
    }

    int a_int = 27;
    int b_int = 72;
    int c_int = a_int + b_int;

    int* a = int2bin[a_int];
    int* b = int2bin[b_int];
    int* c = int2bin[c_int];

    std::vector<double> results;

    for (int pos = 0; pos < 2; pos++) {
        net->feedForward({a[binaryDim - 1 - pos], b[binaryDim - 1 - pos]});
        net->getResults(results);
        std::cout << "Out: ";
        Utils::print(results);
    }

    net->clearMemory();
}

void RNNTest::update() {

}

void RNNTest::draw(sf::RenderWindow &window) {

}
