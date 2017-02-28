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

double Utils::tanhFunction(double x) {
    return tanh(x);
}

double Utils::tanhOutputToDerivative(double x) {
    return (1.0 - x * x); // 1.0 - x * x
}

matrix Utils::sigmoid(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = sigmoid(c);
        }
    }
    return x;
}

matrix Utils::tanhFunction(matrix x) {
    for (auto& r : x.elements) {
        for (auto& c : r) {
            c = tanhFunction(c);
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
            printf("%.16f,", c);
        }
        std::cout << "\b],\n";
        if (ws == 0) ws++;
    }
    //out << "\b\b";
    //std::cout << out.rdbuf();
}