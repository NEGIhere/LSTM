//
// Created by negi on 16.02.17.
//

#include <iostream>
#include <assert.h>
#include "matrix.h"
#include "Utils.h"

matrix::matrix(const unsigned int m, const unsigned int n)
        : elements(m, std::vector<double>(n, 3)) {
}

matrix::matrix(const std::vector<std::vector<double>> &elements)
        : elements(elements) {

}

matrix& matrix::dot(const matrix& other) {
    assert(elements[0].size() == other.elements.size());

    std::vector<std::vector<double>> data(elements.size(), std::vector<double>(other.elements[0].size()));

    for (int row = 0; row < elements.size(); row++) {
        for (int col = 0; col < elements[row].size(); col++) {
            float sum = 0.0f;
            for (int i = 0; i < elements.size(); i++) {
                sum += elements[row][i] * other.elements[i][col];
            }
            data[row][col] = sum;
        }
    }
    elements = data;
    return *this;
}

matrix& matrix::add(const matrix &other) {
    assert(elements.size() == other.elements.size() && elements[0].size() == other.elements[0].size());
    for (int i = 0; i < elements.size(); i++) {
        for (int j = 0; j < elements[0].size(); ++j) {
            elements[i][j] += other.elements[i][j];
        }
    }
    return *this;
}

matrix& matrix::mul(const double num) {
    for (auto& r : elements) {
        for (auto& c : r) {
            c *= num;
        }
    }
    return *this;
}

matrix& matrix::add(const double num) {
    for (auto& r : elements) {
        for (auto& c : r) {
            c += num;
        }
    }
    return *this;
}

matrix& matrix::operator*=(const matrix& other) {
    return dot(other);
}

matrix& matrix::operator*=(const double num) {
    return this->mul(num);
}

matrix &matrix::operator+=(const matrix &other) {
    return add(other);
}


matrix& matrix::operator+=(const double num) {
    return add(num);
}

matrix::~matrix() {
    //delete elements;
}

matrix matrix::random::rand(const unsigned int n, const unsigned int m) {
    matrix mat(n,m);
    for (auto& r : mat.elements) {
        for (auto& c : r) {
            c = Utils::randDouble(0, 1);
        }
    }
    return mat;
}

void matrix::print() {
    //for (int j = 0; j < elements->size(); ++j) {
    //    for (int i = 0; i < elements.get()[j].size(); ++i) {
    //        std::cout << ((std::vector<double>)elements.get()[j][j])[2];

    //    }
    //}
}

