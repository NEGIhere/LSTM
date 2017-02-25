//
// Created by negig on 23.02.2017.
//

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H


#include "utils/matrix.h"
class Model;

class Layer {
public:
    Layer(const unsigned int in, const unsigned int out);
    virtual void feedForward(Model& model, Layer& prevLayer);
    virtual void backPropagate(Model& model, Layer& nextLayer, const unsigned int& step);
    virtual void backPropagate(Model& model, Layer& nextLayer, matrix& XSet, const unsigned int& step);
    matrix W, deltaW, updateW;
    matrix b, deltaB, updateB;
    matrix outputValues;
    std::vector<matrix> gradients;
};

class RNNLayer : public Layer {
public:
    RNNLayer(const unsigned int in, const unsigned int out, int capacity);
    void feedForward(Model& model, Layer& prevLayer);
    void backPropagate(Model& model, Layer& nextLayer, const unsigned int& step);
    matrix memoryW, gradientMemoryW, updateMemoryW;
    std::vector<matrix> prevOutputValues;
    int capacity;
};

class LSTMLayer : public RNNLayer {
public:
    LSTMLayer();
};

class Model {
    friend class Layer;
    friend class RNNLayer;
    friend class LSTMLayer;

public:
    Model(double learningRate, double momentum);
    void addLayer(Layer* layer);
    void train(std::vector<matrix>& XSet, std::vector<matrix>& ySet, double* predicted);
    matrix getOutputValues();
    ~Model();
private:
    std::vector<Layer*> layers;
    matrix* outputValues;
    int recurrentSteps = 0;
    double alpha = 0.3f;

    double eta = 1.5f;
    double overallError = 0;
    double error, recentAverageError = 0;
public:
    double getRecentAverageError() const;

private:

    const double recentAverageSmoothingFactor = 100.0;
};


#endif //LSTM_MODEL_H
