//
// Created by negig on 23.02.2017.
//

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H


#include "utils/matrix.h"

class Layer {
public:
    Layer();
    matrix W, deltaW;
    matrix b, deltaB;
    matrix outputValues;
};

class RNNLayer : public Layer {
public:
    RNNLayer();
    matrix memory;
};

class LSTMLayer : public RNNLayer {
public:
    LSTMLayer();
};

class Model {
public:
    Model();
    void run();
    void setL
private:
    std::vector<Layer> layers;
    //Layer& inputLayer, outputLayer;

    ~Model();
};


#endif //LSTM_MODEL_H
