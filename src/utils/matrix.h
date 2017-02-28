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
    unsigned int numRows, numColumns;

    //explicit matrix(const std::vector<double> &elements);
    explicit matrix(const std::vector<std::vector<double>> &elements);
    explicit matrix(const unsigned int m, const unsigned int n);

    matrix transposed();

    matrix& dot(const matrix& other);
    static matrix mbe(matrix m0, matrix m1); /// Multiply by elements
    static matrix hstack(matrix m0, matrix m1); /// Stack matrices in sequence horizontally (column wise)
    static matrix vstack(matrix m0, matrix m1); /// Stack matrices in sequence vertically (row wise)
    static matrix outer(matrix m0, matrix m1); /// Compute the outer product of two vectors
    matrix& add(const matrix& other);
    matrix& sub(const matrix& other);
    matrix& mul(const double num);
    matrix& add(const double num);
    matrix& sub(const double num);
    matrix& operator+=(const matrix& other);
    matrix& operator-=(const matrix& other);
    matrix& operator+=(const double num);
    matrix& operator-=(const double num);
    matrix& operator*=(const matrix& other);
    matrix& operator*=(const double num);
    std::vector<double>& operator[](const unsigned int i);

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

inline matrix operator-(matrix left, const matrix& right) {
    return left.sub(right);
}

inline matrix operator-(matrix left, const double& num) {
    return left.sub(num);
}

inline matrix operator*(matrix left, const double& num) {
    return left.mul(num);
}

inline matrix operator*(const double& num, matrix right) {
    return right.mul(num);
}

#endif //LSTM_MATRIX_H
