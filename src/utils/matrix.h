//
// Created by negi on 16.02.17.
//

#ifndef LSTM_MATRIX_H
#define LSTM_MATRIX_H


#include <array>
#include <vector>
#include "Utils.h"

struct matrix {
public:
    struct random {
        static matrix rand(const unsigned int n, const unsigned int m);
    };

    std::vector<std::vector<double>> elements;

    //explicit matrix(const std::vector<double> &elements);
    explicit matrix(const std::vector<std::vector<double>> &elements);
    explicit matrix(const unsigned int m, const unsigned int n);

    matrix& dot(const matrix& other);
    matrix& add(const matrix& other);
    matrix& mul(const double num);
    matrix& add(const double num);
    matrix& operator+=(const matrix& other);
    matrix& operator+=(const double num);
    matrix& operator*=(const matrix& other);
    matrix& operator*=(const double num);

    void print();

    virtual ~matrix();

};

inline matrix operator*(matrix left, const matrix& right) {
    return left.dot(right);
}

inline matrix operator+(matrix left, const matrix& right) {
    return left.add(right);
}

inline matrix operator+(matrix left, const double& num) {
    return left.add(num);
}

inline matrix operator*(matrix left, const double& num) {
    return left.mul(num);
}

inline matrix operator*(const double& num, matrix right) {
    return right.mul(num);
}

#endif //LSTM_MATRIX_H
