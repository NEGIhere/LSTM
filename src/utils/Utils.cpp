#include <iostream>
#include "Utils.h"

std::vector<double> Utils::softmax(const std::vector<double>& vec) {
    std::vector<double> out(vec.size());

    int i = 0;

    double sum = 0;
    for(auto& v : vec) {
        sum += exp(v);
    }

    for(auto& v : vec) {
        out[i++] = exp(v) / sum;
    }
    return out;
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

double Utils::transferFunction(double x) {
    return tanh(x);
}

double Utils::transferFunctionDerivative(double x) {
    return 1.0 - tanh(x) * tanh(x); //1.0 - x * x
}

matrix Utils::sigmoid(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = sigmoid(c);
        }
    }
    return x;
}

matrix Utils::transferFunction(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = transferFunction(c);
        }
    }
    return x;}

matrix Utils::transferFunctionDerivative(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = transferFunctionDerivative(c);
        }
    }
    return x;}

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
    out << "[";
    for (auto& r : m.elements) {
        for (int i = 0; i < ws; ++i) {
            out << ' ';
        }
        out << "[";
        for (auto& c : r) {
            out << c << ",";
        }
        out << "\b],\n";
        if (ws == 0) ws++;
    }
    //out << "\b\b";
    std::cout << out.rdbuf();
}