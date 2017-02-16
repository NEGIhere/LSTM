
#ifndef NNTEST_NEURON_H
#define NNTEST_NEURON_H

#include <stdlib.h>
#include <vector>
#include <random>

class Neuron;

struct Connection {
    double weight;
    double deltaWeight;
};

struct Memory {
    std::vector<Connection> outputWeights;
    std::vector<double> outputValues;
    std::vector<double> gradients;
};

typedef std::vector<Neuron> Layer;

class Neuron {

public:
    const int MAX_MEMORY_CELLS = 10;

    Neuron(unsigned int numOutputs, unsigned int totalNeuronsInLayer, unsigned int index);
    void setOutputValue(double val) { outputVal = val; }
    double getOutputValue(void) const { return outputVal; }
    void feedForward(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer);
    void calcOutputGradients(double targetVal);
    void calcOutputGradientsTT(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void calcHiddenGradientsTT(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    void clearMemory();

    virtual ~Neuron();

    std::vector<Connection> outputWeights;
    Memory memory;
    friend class Net;
private:
    void feedForwardMemory(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer);
    static double eta; // net training rate
    static double alpha; // Multiplier or last weight change (momentum)

    double sumDerivativeOutputWeights(const Layer &nextLayer) const;
    double outputVal;
    double index, gradient;
};



#endif //NNTEST_NEURON_H
