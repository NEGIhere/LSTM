#include <iostream>
#include "Utils.h"

// TODO: re-test
std::vector<double> Utils::softmax(const std::vector<double>& vec) {
    std::vector<double> out(vec.size());

    double sum = 0;
    for(auto& v : vec) {
        sum += exp(v);
    }

    for(auto& v : vec) {
        out.push_back(exp(v) / sum);
    }
    return out;
}

int Utils::argmax(const std::vector<double> &vec) {
    int index = 0;
    double max = vec[0];

    for(int i = 0; i < vec.size(); i++) {
        if (max < vec[i]) {
            index = i;
            max = vec[i];
        }
    }
    return index;
}

double Utils::sigmoid(double x) {
    return (1.0 / (1 + exp(-x)));
}

double Utils::sigmoidDerivative(double x) {
    // (exp(x) / ((exp(x) + 1) * (exp(x) + 1)))
    // (1/(1+exp(-x)))*(1-(1/(1+exp(-x))))
    return (1/(1+exp(-x)))*(1-(1/(1+exp(-x))));
}

double Utils::sigmoidOutputToDerivative(double output) {
    return output * (1 - output);
}

matrix Utils::sigmoid(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = sigmoid(c);
        }
    }
    return x;
}

matrix Utils::tanh(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = tanh(c);
        }
    }
    return x;
}

matrix Utils::tanhOutputToDerivative(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = tanhOutputToDerivative(c);
        }
    }
    return x;
}

matrix Utils::sigmoidDerivative(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = sigmoidDerivative(c);
        }
    }
    return x;}

matrix Utils::sigmoidOutputToDerivative(matrix output) {
    for (auto& r : output.elements) {
        for (auto& c : r) {
            c = sigmoidOutputToDerivative(c);
        }
    }
    return output;
}

void Utils::print(const matrix &m) {
    int ws = 0; // whitespaces
    std::stringstream out;
    std::cout << "[";
    for (auto& r : m.elements) {
        for (int i = 0; i < ws; ++i) {
            std::cout << ' ';
        }
        std::cout << "[";
        for (auto& c : r) {
            //out << c << ",";
            if (int(c) == c){
                printf("%.0f, ", c);
            } else {
                printf("%.16f, ", c);
            }
        }
        std::cout << "\b],\n";
        if (ws == 0) ws++;
    }
    //out << "\b\b";
    //std::cout << out.rdbuf();
}

matrix Utils::outer(std::vector<double> v0, std::vector<double> v1) {
    matrix mat(v0.size(), v1.size());
    for (int i = 0; i < mat.numRows; i++) {
        for (int j = 0; j < mat.numColumns; j++) {
            mat[i][j] = v0[i]*v1[j];
        }
    }
    return mat;
}
