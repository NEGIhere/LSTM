
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
        // for randomization
        double factor = 0.7 * pow(numOutputs, 1.0 / (totalNeuronsInLayer - 0));
        double sum = 0, memorySum = 0;

        for (int i = 0; i < numOutputs; i++) {
            outputWeights.push_back(Connection());
            double rw = Utils::randDouble(-0.5, 0.5);
            outputWeights.back().weight = rw;
            sum += rw * rw;
        }

        sum = sqrt(sum);

        for (int i = 0; i < numOutputs; i++) {
            //std::cout << outputWeights[i].weight << "*=" << factor << "/" << sum << "=" << outputWeights[i].weight << std::endl;
            outputWeights[i].weight *= factor / sum;
            /*if (numOutputs == 1) {
                printf("[%f],", outputWeights[i].weight);
            }*/
        }

        for (int i = 0; i < totalNeuronsInLayer - 1; ++i) {
            memory.outputWeights.push_back(Connection());
            memory.outputWeights[i].weight = Utils::randDouble(-0.5, 0.5);
        }
        memory.outputValues.push_back(0);
    }

    this->index = index;
}

Neuron::~Neuron() {
    outputWeights.clear();
}

double Neuron::sumDerivativeOutputWeights(const Layer &nextLayer) const {
    double sum = 0.0;

    for (int i = 0; i < nextLayer.size() - 1; i++) {
        sum += memory.outputWeights[0].weight + outputWeights[i].weight * nextLayer[i].gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (int i = 0; i < prevLayer.size(); i++) {
        Neuron &neuron = prevLayer[i];

        double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputValue() * gradient + alpha * oldDeltaWeight;

        neuron.outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.outputWeights[index].weight += newDeltaWeight;
    }
}

void Neuron::calcOutputGradients(double targetVal) {
    // TODO: should I use SUM 1/2((target_i - out_i)^2) ?
    // error
    double delta = targetVal - outputVal;
    gradient = delta * sigmoidOutputToDerivative(outputVal);
}

void Neuron::calcOutputGradientsTT(double targetVal) {
    double delta = targetVal - outputVal;
    gradient = delta * sigmoidOutputToDerivative(outputVal);
    memory.gradients.push_back(gradient);
    Utils::print(memory.gradients);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDerivativeOutputWeights(nextLayer);
    gradient = dow * sigmoidOutputToDerivative(outputVal);
}

void Neuron::calcHiddenGradientsTT(const Layer &nextLayer) {
    //double dow = sumDerivativeOutputWeights(nextLayer);
    double dow = 0.0;
    double dmw = 0.0;

    for (int i = 0; i < nextLayer.size() - 1; i++) {
        for (int j = 0; j < nextLayer[i].memory.gradients.size(); j++) {
            dow += outputWeights[i].weight * nextLayer[i].memory.gradients[j];
        }
    }

    for (int i = 0; i < memory.outputWeights.size() - 1; i++) {
        //dmw += memory.outputWeights[0].weight * memory.gradients
    }

    gradient = (dmw + dow) * sigmoidOutputToDerivative(outputVal);

    dow = 2;
}

void Neuron::feedForward(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer) {
    double memorySum = 0.0;
    double sum = 0.0;

    if (!isOutputLayer) {
        for (int i = 0; i < currentLayer.size() - 1; i++) {
            memorySum += currentLayer[i].memory.outputValues.back() * memory.outputWeights[i].weight;
            //if (currentLayer.size() == 17 && (index == 1)) std::cout << currentLayer[i].memory.outputValues.back() << "*" << memory.outputWeights[i].weight << "=" << a << " ";
        }
        //if (currentLayer.size() == 17 && index == 1) std::cout << "== " << memorySum << std::endl;
    }

    for (int i = 0; i < prevLayer.size() - 1; i++) { // TODO: BIAS
        sum += prevLayer[i].getOutputValue() * prevLayer[i].outputWeights[index].weight;
        /*if (currentLayer.size() == 2) {
            printf("%.16f*%.16f + ", prevLayer[i].getOutputValue(), prevLayer[i].outputWeights[index].weight);
        }*/
    }

    /*if (currentLayer.size() == 2) {
        printf("\n");
        printf("s%.16f\n", sum);
    }*/
    outputVal = sigmoid(sum + memorySum);

    if (currentLayer.size() == 17) {
        //if (index == -1 || index == 1 || index == 2) printf("ov%d=%.016f ", (int)index, outputVal);
        //std::cout << outputVal << " ";
        //std::cout << sum << "+" << memorySum << " ";

    }
    this->memory.outputValues.push_back(outputVal);
}

void Neuron::clearMemory() {
    memory.outputValues.clear();
    memory.outputValues.push_back(0);
}

double Neuron::sigmoid(double x) {
    return (1.0 / (1 + exp(-x)));
}

double Neuron::sigmoidDerivative(double x) {
    // (exp(x) / ((exp(x) + 1) * (exp(x) + 1)))
    // (1/(1+exp(-x)))*(1-(1/(1+exp(-x))))
    return (1/(1+exp(-x)))*(1-(1/(1+exp(-x))));
}

double Neuron::sigmoidOutputToDerivative(double output) {
    return output * (1 - output);
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - tanh(x) * tanh(x); //1.0 - x * x
}
