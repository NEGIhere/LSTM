#include <assert.h>
#include <math.h>
#include <iostream>
#include "Net.h"

double Net::recentAverageSmoothingFactor = 100.0;

Net::Net(const std::vector<unsigned int> &topology)
        : error(0.0), recentAverageError(0.0) {
    assert(!topology.empty());

    unsigned int numLayers = topology.size();
    for (int i = 0; i < numLayers; i++) {
        Layer layer;

        unsigned int numOutputs = (i == (topology.size() - 1) ? 0 : topology[i + 1]);
        unsigned numNeurons = topology[i] + 1;
        for (unsigned int j = 0; j < numNeurons; j++) {
            layer.push_back(Neuron(numOutputs, numNeurons, j));
        }
        /*if(i == 1) {
            for (int j = 0; j < 16; j++) {
                std::cout << "[";
                for (Connection c : layer[j].memory.outputWeights) {
                    printf("%.16f,", c.weight);
                    //std::cout << c.weight << ",";
                }
                std::cout << "]," << std::endl;
            }
        }*/

        layer.back().setOutputValue(1.0);

        layers.push_back(layer);
    }

}

Net::~Net() {
    layers.clear();
}

void Net::feedForward(const std::vector<double> &inputVals) {
    assert(inputVals.size() == layers[0].size() - 1);

    for (int i = 0; i < inputVals.size(); i++) {
        layers[0][i].setOutputValue(inputVals[i]);
    }

    for (unsigned int i = 1; i < layers.size(); i++) {
        for (unsigned int j = 0; j < layers[i].size() - 1; j++) {
            Layer &prevLayer = layers[i - 1];
            Layer &currentLayer = layers[i];
            layers[i][j].feedForward(prevLayer, currentLayer, i == (layers.size() - 1));
        }
    }

    for (unsigned int i = 1; i < layers.size(); i++) {
        for (unsigned int j = 0; j < layers[i].size() - 1; j++) {
            Layer &prevLayer = layers[i - 1];
            Layer &currentLayer = layers[i];
            layers[i][j].feedForwardMemory(prevLayer, currentLayer, i == (layers.size() - 1));
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals) {
    Layer &outputLayer = layers.back();
    error = 0.0;

    // Root mean square error: sqrt(SUM(i,n, (target[i] - actual[i]) ^ 2) / n)
    for (int n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputValue();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1;
    error = (double) sqrt(error);

    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Gradients for output layer
    for (int i = 0; i < outputLayer.size() - 1; i++) {
        outputLayer[i].calcOutputGradients(targetVals[i]);
    }

    // Gradients for hidden layers
    for (int i = layers.size() - 2; i > 0; i--) {
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];

        for (int j = 0; j < hiddenLayer.size(); j++) {
            hiddenLayer[j].calcHiddenGradients(nextLayer);
        }
    }

    // Update connection weights for layers n .. 1
    for (int i = layers.size() - 1; i > 0; i--) {
        Layer &layer = layers[i];
        Layer &prevLayer = layers[i - 1];

        for (int j = 0; j < layer.size() - 1; j++) {
            layer[j].updateInputWeights(prevLayer);
        }
    }
}

void Net::backPropThroughTimeOutput(const std::vector<double> &targetVals) {
    Layer &outputLayer = layers.back();
    error = 0.0;

    for (int n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputValue();
        error += delta * delta;
    }
    error /= outputLayer.size() - 1;
    error = (double) sqrt(error);

    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

    // Gradients for output layer
    for (int i = 0; i < outputLayer.size() - 1; i++) {
        outputLayer[i].calcOutputGradientsTT(targetVals[i]);
    }
}

void Net::backPropThroughTime(const std::vector<double> &targetVals) {
    // Gradients for hidden layers
    for (unsigned long i = layers.size() - 2; i > 0; i--) {
        Layer &hiddenLayer = layers[i];
        Layer &nextLayer = layers[i + 1];

        for (int j = 0; j < hiddenLayer.size(); j++) {
            hiddenLayer[j].calcHiddenGradientsTT(nextLayer);
        }
    }
}

void Net::clearMemory() {
    for (unsigned int i = 1; i < layers.size(); i++) {
        for (unsigned int j = 0; j < layers[i].size() - 1; j++) {
            layers[i][j].clearMemory();
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();

    for (int i = 0; i < layers.back().size() - 1; i++) {
        resultVals.push_back(layers.back()[i].getOutputValue());
    }
}

void Net::getWeights(std::vector<double> &weightVals, bool bias) {
    weightVals.clear();
    if (bias) {
        for (unsigned int i = 0; i < layers.size() - 1; i++) {
            for (unsigned int j = 0; j < layers[i].size(); j++) {
                for (unsigned int k = 0; k < layers[i + 1].size(); k++) {
                    weightVals.push_back(layers[i][j].outputWeights[k].weight);
                }
            }
        }
    } else {
        for (unsigned int i = 0; i < layers.size() - 1; i++) {
            for (unsigned int j = 0; j < layers[i].size() - 1; j++) {
                for (unsigned int k = 0; k < layers[i + 1].size() - 1; k++) {
                    weightVals.push_back(layers[i][j].outputWeights[k].weight);
                }
            }
        }
    }
}

void Net::setWeights(const std::vector<double> &weightVals, bool bias) {
    int numConnections = 0;
    if (bias) {
        for (unsigned int i = 0; i < layers.size() - 1; i++) {
            for (unsigned int j = 0; j < layers[i].size(); j++) {
                for (unsigned int k = 0; k < layers[i + 1].size(); k++) {
                    layers[i][j].outputWeights[k].weight = double(weightVals[numConnections++]);
                }
            }
        }
    } else {
        for (unsigned int i = 0; i < layers.size() - 1; i++) {
            for (unsigned int j = 0; j < layers[i].size() - 1; j++) {
                for (unsigned int k = 0; k < layers[i + 1].size() - 1; k++) {
                    layers[i][j].outputWeights[k].weight = double(weightVals[numConnections++]);
                }
            }
        }
    }
}

int Net::getConnectionsCount(bool bias) const {
    int num = 0;
    if (bias) {
        for (int i = 0; i < layers.size() - 1; i++) {
            num += layers[i].size() * layers[i + 1].size();
        }
    } else {
        for (int i = 0; i < layers.size() - 1; i++) {
            num += (layers[i].size() - 1) * (layers[i + 1].size() - 1);
        }
    }
    return num;
}

const std::vector<Layer> &Net::getLayers() const {
    return layers;
}

bool Net::isFirstRunning() const {
    return firstRunning;
}