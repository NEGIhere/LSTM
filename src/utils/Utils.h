
#ifndef NNLIFE_UTILS_H
#define NNLIFE_UTILS_H

#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iterator>
#include <sstream>
#include "matrix.h"
#include <iostream>

#define PI 3.14159265359
/// Multiply by this value
#define rad2deg (1 / PI * 180.0)
/// Multiply by this value
#define deg2rad (1 / 180.0 * PI)

struct matrix;

class Utils {
public:
    static double randDouble(double first, double last) {
        return first + (rand() / double(RAND_MAX)) * (last - first);
    }

    static int randInt(int first, int last) {
        return first + (rand() % (last - first + 1));
    }

    inline double radToDeg(float rad) {
        return rad / PI * 180.0;
    }

    inline double degToRad(float deg) {
        return deg / 180.0 * PI;
    }

    static std::vector<double> softmax(const std::vector<double>& vec);
    static int argmax(const std::vector<double>& vec);
    static matrix outer(std::vector<double> v0, std::vector<double> v1);
    template<typename Base, typename T> static inline bool instanceOf(const T *) {
        return std::is_base_of<Base, T>::value;
    }

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
        std::cout << o << std::endl;
    }

    static void print(const matrix& m);

    static double sigmoid(double x);
    static matrix sigmoid(matrix x);
    static inline double tanh(double x) { return std::tanh(x); };
    static matrix tanh(matrix x);
    static inline double tanhOutputToDerivative(double x) { return (1.0 - x * x); };
    static matrix tanhOutputToDerivative(matrix x);
    static double sigmoidDerivative(double x);
    static matrix sigmoidDerivative(matrix x);
    static double sigmoidOutputToDerivative(double output);
    static matrix sigmoidOutputToDerivative(matrix output);
};

#endif //NNLIFE_UTILS_H
