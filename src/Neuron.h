
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
    double outputVal;
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
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

    virtual ~Neuron();

    std::vector<Connection> outputWeights;
    //std::vector<Memory> memoryCells;
    Memory memory;
private:
    static double eta; // net training rate
    static double alpha; // Multiplier or last weight change (momentum)
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double sigmoid(double x);
    double sigmoidDerivative(double x);

    double sumDerivativeOutputWeights(const Layer &nextLayer) const;
    double outputVal;
    double index, gradient;
};



#endif //NNTEST_NEURON_H
