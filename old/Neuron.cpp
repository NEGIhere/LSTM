
#include <math.h>
#include <iostream>
#include <assert.h>
#include "Neuron.h"
#include "../src/utils/Utils.h"
#include "Net.h"
#include "../src/utils/matrix.h"

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
    gradient = delta * Utils::sigmoidOutputToDerivative(outputVal);
}

void Neuron::calcOutputGradientsTT(double targetVal) {
    double delta = targetVal - outputVal;
    gradient = delta * Utils::sigmoidOutputToDerivative(outputVal);
    memory.gradients.push_back(gradient);
    //Utils::print(memory.gradients);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDerivativeOutputWeights(nextLayer);
    gradient = dow * Utils::sigmoidOutputToDerivative(outputVal);
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

    gradient = (dmw + dow) * Utils::sigmoidOutputToDerivative(outputVal);

    dow = 2;
}

void Neuron::feedForward(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer) {
    double sum = 0.0;

    for (int i = 0; i < prevLayer.size() - 1; i++) { // TODO: BIAS
        sum += prevLayer[i].getOutputValue() * prevLayer[i].outputWeights[index].weight;
        if (currentLayer.size() == 17) {
            //printf("%.16f*%.16f + ", prevLayer[i].getOutputValue(), prevLayer[i].outputWeights[index].weight);
        }
    }

    outputVal = sum;
}

void Neuron::feedForwardMemory(const Layer &prevLayer, const Layer &currentLayer, bool isOutputLayer) {
    double memorySum = 0.0;

    if (!isOutputLayer) {
        for (int i = 0; i < currentLayer.size() - 1; i++) {
            double d = currentLayer[i].memory.outputValues.back();
            double weight = memory.outputWeights[i].weight;
            memorySum += d * weight;
            if (currentLayer.size() == 17) printf("%.16f*%.16f + ", d, weight);
        }
        //if (currentLayer.size() == 17 && index == 1) std::cout << "== " << memorySum << std::endl;
    }

    printf("= %.16f\n", memorySum);

    outputVal = Utils::sigmoid(outputVal + memorySum);

    this->memory.outputValues.push_back(outputVal);
}

void Neuron::clearMemory() {
    memory.outputValues.clear();
    memory.outputValues.push_back(0);
}
