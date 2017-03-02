//
// Created by negi on 16.02.17.
//

#include <iostream>
#include <assert.h>
#include "matrix.h"
#include "Utils.h"

matrix::matrix(const unsigned int m, const unsigned int n)
        : elements(m, std::vector<double>(n, 0)), numRows(m), numColumns(n) {
}

matrix::matrix(const std::vector<std::vector<double>> &elements)
        : elements(elements), numRows((int)elements.size()), numColumns((int)elements[0].size()) {
}

matrix matrix::dot(const matrix& other) {
    assert(numColumns == other.numRows);

    std::vector<std::vector<double>> data(numRows, std::vector<double>(other.numColumns));

    for (int row = 0; row < numRows; row++) {
        for (int col = 0; col < other.numColumns; col++) {
            double sum = 0.0;
            for (int i = 0; i < other.numRows; i++) {
                double a = elements[row][i];
                double b = other.elements[i][col];
                sum += a * b;
            }
            data[row][col] = sum;
        }
    }

    elements = data;
    numRows = (int)data.size();
    numColumns = (int)data[0].size();
    return *this;
}

matrix& matrix::add(const matrix &other) {
    assert(numRows == other.numRows && numColumns == other.numColumns);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; ++j) {
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

/// Multyply element by element
matrix matrix::mbe(matrix m0, matrix m1) {
    assert(m0.numRows == m1.numRows && m0.numColumns == m1.numColumns);

    for (int i = 0; i < m0.numRows; i++) {
        for (int j = 0; j < m0.numColumns; ++j) {
            m0.elements[i][j] *= m1.elements[i][j];
        }
    }
    return m0;
}

matrix matrix::hstack(matrix m0, matrix m1) {
    assert(m0.numRows == m1.numRows);
    matrix m2(m0.numRows, m0.numColumns + m1.numColumns);
    for (int i = 0; i < m2.numRows; i++) {
        int j = 0;
        for (; j < m0.numColumns; j++) {
            m2.elements[i][j] = m0.elements[i][j];
        }
        for (; j < m2.numColumns; j++) {
            m2.elements[i][j] = m1.elements[i][j - m0.numColumns];
        }
    }
    return m2;
}

// TODO: create 'vstack'
matrix matrix::vstack(matrix m0, matrix m1) {
    assert(m0.numColumns == m1.numColumns);
    matrix m2(m0.numRows + m1.numRows, m0.numColumns);

    return m2;
}

// TODO: create 'outer'
matrix matrix::outer(matrix m0, matrix m1) {

    return m0;
}

matrix& matrix::add(const double num) {
    for (auto& r : elements) {
        for (auto& c : r) {
            c += num;
        }
    }
    return *this;
}

matrix& matrix::sub(const matrix &other) {
    assert(numRows == other.numRows && numColumns == other.numColumns);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; ++j) {
            elements[i][j] -= other.elements[i][j];
        }
    }
    return *this;
}

matrix& matrix::sub(const double num) {
    for (auto& r : elements) {
        for (auto& c : r) {
            c -= num;
        }
    }
    return *this;
}

matrix matrix::operator*=(const matrix& other) {
    return (*this).dot(other);
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

matrix &matrix::operator-=(const matrix &other) {
    return sub(other);
}

matrix &matrix::operator-=(const double num) {
    return sub(num);
}

std::vector<double>& matrix::operator[](const unsigned int i) {
    return elements[i];
}

matrix matrix::transposed() {
    matrix transposed(numColumns, numRows);

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            transposed.elements[j][i] = elements[i][j];
        }
    }

    return transposed;
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

matrix::~matrix() {
    //delete elements;
}
