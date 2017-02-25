//
// Created by negig on 23.02.2017.
//

#include <assert.h>
#include "Model.h"

Model::Model(double learningRate, double momentum) {
    eta = learningRate;
    alpha = momentum;
    outputValues = nullptr;
}

void Model::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void Model::train(std::vector<matrix>& XSet, std::vector<matrix>& ySet, double* predicted, unsigned int predictedCount) {
    const unsigned int size = (unsigned int)XSet.size();
    assert(size == ySet.size());

    for (int i = 0; i < size; i++) {
        matrix& X = XSet[i];
        matrix& y = ySet[i];

        Layer &inputLayer = *layers[0];
        inputLayer.outputValues = X;// * inputLayer.W;

        for (int j = 1; j < layers.size(); j++) {
            Layer &prevLayer = *layers[j - 1];
            Layer &currLayer = *layers[j];
            currLayer.feedForward(*this, prevLayer);
        }
        Layer &outputLayer = *layers.back();
        matrix l2Error = y - outputLayer.outputValues;
        outputLayer.gradients.push_back(matrix::mbe(l2Error, Utils::sigmoidOutputToDerivative(outputLayer.outputValues)));
        //Utils::print(l2Error);

        error = 0.0;
        for (int n = 0; n < outputLayer.outputValues.numColumns; n++) {
            double delta = y[0][n] - outputLayer.outputValues[n][0];
            error += delta * delta;
        }
        error /= outputLayer.outputValues.numColumns;
        error = (double) sqrt(error);

        recentAverageError =
                (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

        outputValues = &outputLayer.outputValues;

        predicted[i] = (*outputValues)[0][0];
    }

    for (unsigned int step = 0; step < size; step++) {
        for (int i = (int)layers.size() - 2; i >= 0; i--) {
            Layer &currLayer = *layers[i];
            Layer &nextLayer = *layers[i + 1];
            if (i == 0) {
                currLayer.backPropagate(*this, nextLayer, XSet[size - step - 1], size - step - 1);
            } else {
                currLayer.backPropagate(*this, nextLayer, size - step - 1);
            }
        }
    }

    recurrentSteps = 0;
    for(std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
        Layer* layer = *it;
        layer->W += layer->updateW * alpha;
        layer->b += layer->updateB * alpha;
        layer->updateW *= 0;
        layer->updateB *= 0;

        RNNLayer* rnnLayer;
        if ((rnnLayer = dynamic_cast<RNNLayer*>(*it)) != nullptr) {
            rnnLayer->memoryW += rnnLayer->updateMemoryW * alpha;
            rnnLayer->updateMemoryW *= 0;

            rnnLayer->prevOutputValues.clear();
            rnnLayer->prevOutputValues.push_back(matrix(rnnLayer->outputValues.numRows, rnnLayer->outputValues.numColumns));
            rnnLayer->outputValues *= 0;
            rnnLayer->gradientMemoryW *= 0;
        }
        (*it)->gradients.clear();
    }
}

matrix Model::getOutputValues() {
    return *outputValues;
}

Layer::Layer(const unsigned int in, const unsigned int out) :
        W(2.0 * matrix::random::rand(in,out) - 1.0), deltaW(in,out), b(2.0 * matrix::random::rand(1,out) - 1.0), deltaB(1,out),
        outputValues(1,in), updateW(W.numRows, W.numColumns), updateB(b.numRows, b.numColumns) {
}

void Layer::feedForward(Model &model, Layer &prevLayer) {
    outputValues = Utils::sigmoid(prevLayer.outputValues*prevLayer.W + 1.0*prevLayer.b);
}

void Layer::backPropagate(Model &model, Layer &nextLayer, const unsigned int& step) {

}

void Layer::backPropagate(Model &model, Layer &nextLayer, matrix &XSet, const unsigned int &step) {
    matrix l0 = XSet;
    updateW += model.eta * l0.transposed()*((RNNLayer&)nextLayer).gradientMemoryW;
    updateB += (1.0 * matrix::mbe(b, ((RNNLayer&)nextLayer).gradientMemoryW));
}

RNNLayer::RNNLayer(const unsigned int in, const unsigned int out, int capacity) :
        Layer::Layer(in, out), memoryW(2.0 * matrix::random::rand(in, in) - 1.0), capacity(capacity), updateMemoryW(memoryW.numRows, memoryW.numColumns), gradientMemoryW(1,in) {
    prevOutputValues.push_back(matrix(1, in));
}

void RNNLayer::feedForward(Model& model, Layer &prevLayer) {
    outputValues = Utils::sigmoid(prevLayer.outputValues*prevLayer.W + prevOutputValues.back()*memoryW + 1.0*prevLayer.b);
    prevOutputValues.push_back(outputValues);
    model.recurrentSteps++;
}

void RNNLayer::backPropagate(Model &model, Layer &nextLayer, const unsigned int& step) {
    RNNLayer* rnnLayer;
    Layer* layer = &nextLayer;
    matrix outputGradient(1,1);
    if ((rnnLayer = dynamic_cast<RNNLayer*>(layer)) != nullptr) {
        outputGradient = rnnLayer->gradientMemoryW;
    } else {
        outputGradient = nextLayer.gradients[step];

    }
    matrix& l1 = prevOutputValues[step + 1];
    matrix& pl1 = prevOutputValues[step];

    updateW += model.eta * l1.transposed() * outputGradient;
    updateB += (1.0 * matrix::mbe(b, outputGradient));
    matrix hiddenGradient = matrix::mbe((gradientMemoryW*memoryW.transposed() + outputGradient*W.transposed()), Utils::sigmoidOutputToDerivative(l1));
    updateMemoryW += model.eta * pl1.transposed()*hiddenGradient;
    gradientMemoryW = hiddenGradient;

}

Model::~Model() {
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
    layers.clear();
}

double Model::getRecentAverageError() const {
    return recentAverageError;
}

