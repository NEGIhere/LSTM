
#ifndef NNLIFE_UTILS_H
#define NNLIFE_UTILS_H

#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iterator>
#include <sstream>
#include "matrix.h"

#define PI 3.14159265359f
#define rad2deg (1 / PI * 180.0f)
#define deg2rad (1 / 180.0f * PI)

struct matrix;

class Utils {
public:
    static double randDouble(double first, double last) {
        return first + (rand() / double(RAND_MAX)) * (last - first);
    }

    static int randInt(int first, int last) {
        return first + (rand() % (last - first + 1));
    }

    inline float radToDeg(float rad) {
        return rad / PI * 180.f;
    }

    inline float degToRad(float deg) {
        return deg / 180.0f * PI;
    }

    static std::vector<double> softmax(const std::vector<double>& vec);

    template<typename T> static void print(const std::vector<T>& vec) {
        if (!vec.empty()) {
            std::cout << '[';
            for (auto& it : vec) {
                printf("%.16f,", it);
            }
            std::cout << "\b\b]" << std::endl;
        }
    }

    template<typename T> static void print(const T& o) {
        std::cout << ">> " << o << std::endl;
    }

    static void print(const matrix& m);

    static double randomWeight(void) { return rand() / double(RAND_MAX); }

    static double sigmoid(double x);
    static matrix sigmoid(matrix x);
    static double transferFunction(double x);
    static matrix transferFunction(matrix x);
    static double transferFunctionDerivative(double x);
    static matrix transferFunctionDerivative(matrix x);
    static double sigmoidDerivative(double x);
    static matrix sigmoidDerivative(matrix x);
    static double sigmoidOutputToDerivative(double output);
    static matrix sigmoidOutputToDerivative(matrix output);
};

#endif //NNLIFE_UTILS_H
