//
// Created by negi on 26.02.17.
//

#include <values.h>
#include "DataSet.h"

DataSet::DataSet(const unsigned int size) {
    setX.reserve(size);
    setY.reserve(size);
    setPredicted.reserve(size);

    setXMin = setYMin = MAXDOUBLE;
    setXMax = setYMax = MINDOUBLE;
}

void DataSet::add(const double x, const double y) {
    setX.push_back(x);
    setY.push_back(y);

    if (setXMin > x) {
        setXMin = x;
    } else if (setXMax < x) {
        setXMax = x;
    }

    if (setYMin > y) {
        setYMin = y;
    } else if (setYMax < y) {
        setYMax = y;
    }
}

void DataSet::normalize() {
    for (int i = 0; i < setX.size(); i++) {
        setX[i] = (setX[i] - setXMin) / (setXMax - setXMin);
        setY[i] = (setY[i] - setYMin) / (setYMax - setYMin);
    }
}

double &DataSet::operator[](const unsigned int i) {
    return setY[i];
}

double DataSet::unpackX(const unsigned int& i) {
    return setX[i] * (setXMax - setXMin) + setXMin;
}

double DataSet::unpackY(const unsigned int& i) {
    return setY[i] * (setYMax - setYMin) + setYMin;
}

double DataSet::unpack(double value) {
    return value * (setYMax - setYMin) + setYMin;
}
