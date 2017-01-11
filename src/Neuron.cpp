
#include <math.h>
#include <iostream>
#include <assert.h>
#include "Neuron.h"
#include "Utils.h"
#include "Net.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned int numOutputs, unsigned int totalNeuronsInLayer, unsigned int index) {
    if (numOutputs != 0) {
        double factor = 0.7 * pow(numOutputs, 1.0 / (totalNeuronsInLayer - 0));
        double sum = 0;

        for(int i = 0; i < numOutputs; i++) {
            outputWeights.push_back(Connection());

            double rw = Utils::randomDouble(-0.5, 0.5);

            outputWeights.back().weight = rw;
            sum += rw * rw;
        }

        sum = sqrt(sum);

        for(int i = 0; i < numOutputs; i++) {
            outputWeights[i].weight *= factor / sum;
        }

        memory.outputWeights.push_back(Connection());
        memory.outputWeights[0].weight = Utils::randomDouble(-0.5, 0.5);
    }

    this->index = index;
}

Neuron::~Neuron() {
    outputWeights.clear();
}

double Neuron::sumDerivativeOutputWeights(const Layer &nextLayer) const {
    double sum = 0.0;

    for(int i = 0; i < nextLayer.size() - 1; i++) {
        sum += outputWeights[i].weight * nextLayer[i].gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for(int i = 0; i < prevLayer.size(); i++) {
        Neuron &neuron = prevLayer[i];

        double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputValue() * gradient + alpha * oldDeltaWeight;

        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - outputVal;
    gradient = delta * transferFunctionDerivative(outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDerivativeOutputWeights(nextLayer);
    gradient = dow * transferFunctionDerivative(outputVal);
}

void Neuron::feedForward(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer) {
    double memorySum = 0.0;
    double sum = 0.0;

    if (!isOutputLayer) {
        if (Net::memoryConnectionType == MANY_TO_MANY) {
            for (int i = 0; i < currentLayer.size(); i++) {
                memorySum += currentLayer[i].memory.outputVal *
                             currentLayer[i].memory.outputWeights[index].weight; //currentLayer[i].getMemoryOutputValue() * currentLayer[i].outputWeights[index].weight;
            }
        } else {
            memorySum = memory.outputVal * memory.outputWeights[0].weight;
        }
    }


    for(int i = 0; i < prevLayer.size(); i++) {
        sum += prevLayer[i].getOutputValue() * prevLayer[i].outputWeights[index].weight;
    }

    outputVal = transferFunction(memorySum + sum);

    this->memory.outputVal = outputVal;
}

double Neuron::sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

double Neuron::sigmoidDerivative(double x) {
    return (exp(x) / (exp(x) + 1) * (exp(x) + 1));
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - tanh(x) * tanh(x); //1.0 - x * x
}
