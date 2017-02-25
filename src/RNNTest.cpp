//
// Created by negi on 11.02.17.
//

#include <iostream>
#include <cstring>
#include <math.h>
#include "RNNTest.h"
#include "utils/Utils.h"
#include "Model.h"

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

#undef MODEL
#define MODEL

#ifdef MODEL
    Model model = Model(1.5, 0.3);
    model.addLayer(new Layer(2,16));
    model.addLayer(new RNNLayer(16,1, 8));
    model.addLayer(new Layer(1,1));

    for (int i = 0; i < 50000; i++) {
        int a_int = Utils::randInt(0, (int) ceil(largestNum / 2.0));
        int b_int = Utils::randInt(0, (int) ceil(largestNum / 2.0));
        int c_int = a_int + b_int;

        int *a = int2bin[a_int];
        int *b = int2bin[b_int];
        int *c = int2bin[c_int];
        double *d = new double[binaryDim];

        std::vector<matrix> XSet;
        XSet.reserve((unsigned long)binaryDim);
        std::vector<matrix> ySet;
        ySet.reserve((unsigned long)binaryDim);

        for (int pos = 0; pos < binaryDim; pos++) {
            matrix X = matrix({{a[binaryDim - 1 - pos], b[binaryDim - 1 - pos]}});
            matrix y = matrix(1,1);
            y[0][0] = c[binaryDim - 1 - pos];
            XSet.push_back(X);
            ySet.push_back(y);
        }

        model.train(XSet, ySet, d);

        for (int j = 0; j < binaryDim / 2; j++) {
            std::swap(d[j], d[binaryDim - j - 1]);
        }

        if (i % 1000 == 0) {
            Utils::print(std::string("Error:") + std::to_string(model.getRecentAverageError()));
            Utils::print(std::string("Pred:") + std::to_string(std::round(d[0])) + std::to_string(std::round(d[1])) + std::to_string(std::round(d[2])) + std::to_string(std::round(d[3])) + std::to_string(std::round(d[4])) + std::to_string(std::round(d[5])) + std::to_string(std::round(d[6])) + std::to_string(std::round(d[7])));
            Utils::print(std::string("True:") + std::to_string(c[0]) + std::to_string(c[1]) + std::to_string(c[2]) + std::to_string(c[3]) + std::to_string(c[4]) + std::to_string(c[5]) + std::to_string(c[6]) + std::to_string(c[7]));
            int out = 0;
            for (int j = binaryDim - 1; j >= 0; j--) {
                out += d[j]*pow(2, binaryDim-j-1);
            }
            Utils::print(std::to_string(a_int) + " + " + std::to_string(b_int) + " = " + std::to_string(out));
        }
    }
#else
    std::vector<double> results;

    matrix s0 = 2.0 * matrix::random::rand(2,16) - 1.0;
    matrix b0 = 2.0 * matrix::random::rand(1,16) - 1.0;
    //double b0 = Utils::randDouble(-1, 1);
    //double b1 = Utils::randDouble(-1, 1);
    matrix s1 = 2.0 * matrix::random::rand(16,1) - 1.0;
    matrix b1 = 2.0 * matrix::random::rand(1,1) - 1.0;
    matrix sm = 2.0 * matrix::random::rand(16,16) - 1.0;

    const double alpha = 0.3f;
    const double eta = 1.5f;

    double overallError = 0;
    double error, recentAverageError = 0;
    const double recentAverageSmoothingFactor = 100.0;

    srand(20);

    for (int i = 0; i < 50000; i++) {
        int a_int = Utils::randInt(0, (int)ceil(largestNum/2.0));
        int b_int = Utils::randInt(0, (int)ceil(largestNum/2.0));
        int c_int = a_int + b_int;

        int* a = int2bin[a_int];
        int* b = int2bin[b_int];
        int* c = int2bin[c_int];
        int* predicted = new int[binaryDim];

        matrix s0Update = matrix(2, 16);
        matrix b0Update = matrix(1, 16);
        matrix smUpdate = matrix(16, 16);
        matrix s1Update = matrix(16, 1);
        matrix b1Update = matrix(1, 1);
        std::vector<matrix> l1Values;
        std::vector<matrix> l2Deltas;

        l1Values.push_back(matrix(1,16));
        matrix futureL1Delta = matrix(1,16);

        for (int pos = 0; pos < binaryDim; pos++) {;
            matrix X = matrix({{a[binaryDim - 1 - pos], b[binaryDim - 1 - pos]}});
            matrix y = matrix(1,1);
            y[0][0] = c[binaryDim - 1 - pos];

            matrix l1 = Utils::sigmoid(X*s0 + l1Values.back()*sm + 1.0*b0);
            matrix l2 = Utils::sigmoid(l1*s1 + 1.0*b1);

            matrix l2Error = y - l2;
            l2Deltas.push_back(matrix::mbe(l2Error, Utils::sigmoidOutputToDerivative(l2)));

            //overallError += std::abs(l2Error.elements[0][0]);
            error = 0.0;
            for (int n = 0; n < l2.numColumns; n++) {
                double delta = y[0][n] - l2[n][0];
                error += delta * delta;
            }
            error /= l2.numColumns;
            error = (double)sqrt(error);

            recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

            double &x = l2[0][0];
            predicted[binaryDim - pos - 1] = (int)std::round(x);
            l1Values.push_back(l1);
        }

        for (int pos = 0; pos < binaryDim; pos++) {
            matrix l0 = matrix({{a[pos],b[pos]}});
            matrix l1 = l1Values[binaryDim - pos];
            matrix prevL1 = l1Values[binaryDim - pos - 1];

            matrix l2Delta = l2Deltas[binaryDim - pos - 1];
            s1Update += eta * l1.transposed() * l2Delta;
            b1Update += (1.0 * matrix::mbe(b1, l2Delta));

            matrix l1Delta = matrix::mbe((futureL1Delta*sm.transposed() + l2Delta*s1.transposed()), Utils::sigmoidOutputToDerivative(l1));

            futureL1Delta = l1Delta;

            s0Update += eta * l0.transposed()*l1Delta;
            //Utils::print(l0.transposed());

            b0Update += (1.0 * matrix::mbe(b0, l1Delta));
            smUpdate += eta * prevL1.transposed()*l1Delta;

        }
        //Utils::print("\n");

        s0 += s0Update * alpha;
        b0 += b0Update * alpha;

        sm += smUpdate * alpha;

        s1 += s1Update * alpha;
        b1 += b1Update * alpha;

        s0Update *= 0;
        s1Update *= 0;
        smUpdate *= 0;
        b0Update *= 0;
        b1Update *= 0;

        if (i % 1000 == 0) {
            Utils::print(std::string("Error:") + std::to_string(recentAverageError));
            Utils::print(std::string("Pred:") + std::to_string(predicted[0]) + std::to_string(predicted[1]) + std::to_string(predicted[2]) + std::to_string(predicted[3]) + std::to_string(predicted[4]) + std::to_string(predicted[5]) + std::to_string(predicted[6]) + std::to_string(predicted[7]));
            Utils::print(std::string("True:") + std::to_string(c[0]) + std::to_string(c[1]) + std::to_string(c[2]) + std::to_string(c[3]) + std::to_string(c[4]) + std::to_string(c[5]) + std::to_string(c[6]) + std::to_string(c[7]));
            int out = 0;
            for (int j = binaryDim - 1; j >= 0; j--) {
                out += predicted[j]*pow(2, binaryDim-j-1);
            }
            Utils::print(std::to_string(a_int) + " + " + std::to_string(b_int) + " = " + std::to_string(out));
        }
    }
#endif
}

void RNNTest::update() {

}

void RNNTest::draw(sf::RenderWindow &window) {

}
