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
    int* d = new int[binaryDim];

    std::vector<double> results;

    for (int i = 0; i < 1; i++) {
        for (int pos = 0; pos < 2; pos++) {
            printf("ITERATION - %d\n", pos);
            int x0 = a[binaryDim - 1 - pos];
            int x1 = b[binaryDim - 1 - pos];
            net->feedForward({x0, x1});
            net->getResults(results);
            std::cout << "Out: ";
            Utils::print(results);
            d[binaryDim - pos - 1] = (int)round(results[0]);

            int Y = c[binaryDim - 1 - pos];
            net->backPropThroughTimeOutput({Y});
        }

        for (int pos = 0; pos < 2; pos++) {
            printf("BP ITERATION - %d\n", pos);
            net->backPropThroughTime({0});
        }

        net->clearMemory();
    }
}

void RNNTest::update() {

}

void RNNTest::draw(sf::RenderWindow &window) {

}
