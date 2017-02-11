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
