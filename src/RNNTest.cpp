//
// Created by negi on 11.02.17.
//

#include <iostream>
#include <cstring>
#include <math.h>
#include "RNNTest.h"
#include "utils/Utils.h"

RNNTest::RNNTest() {
    int binaryDim = 8;
    int largestNum = (int)pow(2, binaryDim);
    int* int2bin[largestNum + 1];

    for (int i = 0; i < largestNum + 1; i++) {
        int2bin[i] = new int[binaryDim];

        int num = i;
        for (int j = binaryDim - 1; j >= 0; --j) {
            int2bin[i][j] = num % 2;
            num /= 2;
        }
    }

    std::vector<double> results;

    matrix s0 = 2.0 * matrix::random::rand(2,16) - 1.0;
    matrix sm = 2.0 * matrix::random::rand(16,16) - 1.0;
    matrix s1 = 2.0 * matrix::random::rand(16,1) - 1.0;

    const double alpha = 0.1f;

    for (int i = 0; i < 50000; i++) {
        double overallError = 0;

        int a_int = Utils::randInt(0, (int)ceil(largestNum/2.0));
        int b_int = Utils::randInt(0, (int)ceil(largestNum/2.0));
        int c_int = a_int + b_int;

        int* a = int2bin[a_int];
        int* b = int2bin[b_int];
        int* c = int2bin[c_int];
        int* d = new int[binaryDim];

        matrix s0Update = matrix(2, 16);
        matrix smUpdate = matrix(16, 16);
        matrix s1Update = matrix(16, 1);
        std::vector<matrix> l1Values;
        std::vector<matrix> l2Deltas;

        l1Values.push_back(matrix(1,16));
        matrix futureL1Delta = matrix(1,16);

        for (int pos = 0; pos < binaryDim; pos++) {;
            matrix X = matrix({{a[binaryDim - 1 - pos], b[binaryDim - 1 - pos]}});
            matrix y = matrix(1,1);
            y.elements[0][0] = c[binaryDim - 1 - pos];

            matrix l1 = Utils::sigmoid(X*s0 + l1Values.back()*sm);
            matrix l2 = Utils::sigmoid(l1*s1);

            matrix l2Error = y - l2;

            l2Deltas.push_back(matrix::mbe(l2Error, Utils::sigmoidOutputToDerivative(l2)));

            overallError += std::abs(l2Error.elements[0][0]);
            d[binaryDim - pos - 1] = (int)std::round(l2.elements[0][0]);
            l1Values.push_back(l1);
        }

        for (int pos = 0; pos < binaryDim - 0; pos++) {
            matrix l0 = matrix({{a[pos],b[pos]}});
            matrix l1 = l1Values[binaryDim - pos];
            matrix prevL1 = l1Values[binaryDim - pos - 1];
            matrix l2Delta = l2Deltas[binaryDim - pos - 1];

            matrix l1Delta = matrix::mbe((futureL1Delta*sm.transposed() + l2Delta*s1.transposed()), Utils::sigmoidOutputToDerivative(l1));

            s1Update += l1.transposed()*l2Delta;
            smUpdate += prevL1.transposed()*l1Delta;
            s0Update += l0.transposed()*l1Delta;

            futureL1Delta = l1Delta;
        }

        s0 += s0Update * alpha;
        s1 += s1Update * alpha;
        sm += smUpdate * alpha;

        s0Update *= 0;
        s1Update *= 0;
        smUpdate *= 0;

        if (i % 1000 == 0) {
            Utils::print(std::string("Error:") + std::to_string(overallError));
            Utils::print(std::string("Pred:") + std::to_string(d[0]) + std::to_string(d[1]) + std::to_string(d[2]) + std::to_string(d[3]) + std::to_string(d[4]) + std::to_string(d[5]) + std::to_string(d[6]) + std::to_string(d[7]));
            Utils::print(std::string("True:") + std::to_string(c[0]) + std::to_string(c[1]) + std::to_string(c[2]) + std::to_string(c[3]) + std::to_string(c[4]) + std::to_string(c[5]) + std::to_string(c[6]) + std::to_string(c[7]));
            int out = 0;
            for (int j = binaryDim - 1; j >= 0; j--) {
                out += d[j]*pow(2, binaryDim-j-1);
            }
            Utils::print(std::to_string(a_int) + " + " + std::to_string(b_int) + " = " + std::to_string(out));
        }
    }
    /*
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
    */
}

void RNNTest::update() {

}

void RNNTest::draw(sf::RenderWindow &window) {

}
