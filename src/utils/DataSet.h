//
// Created by negi on 26.02.17.
//

#ifndef LSTM_TRAINSET_H
#define LSTM_TRAINSET_H


#include <vector>

class DataSet {
public:
    DataSet(const unsigned int size);
    std::vector<double> setX, setY, setPredicted;
    void add(const double x, const double y);
    void normalize();
    double& operator[](const unsigned int i);
    double unpackX(const unsigned int& i);
    double unpackY(const unsigned int& i);
    double unpack(double value);
private:
    double setXMin, setXMax, setYMin, setYMax;
};


#endif //LSTM_TRAINSET_H
