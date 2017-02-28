//
// Created by negig on 23.02.2017.
//

#include <assert.h>
#include "Model.h"

Model::Model(double learningRate, double momentum) {
    eta = learningRate;
    mu = (1.0 - momentum);
    outputValues = nullptr;
}

void Model::addLayer(Layer* layer) {
    layers.push_back(layer);
}

void Model::train(std::vector<matrix>& XSet, std::vector<matrix>& ySet, double* predicted) {
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
        matrix outputLayerError = y - outputLayer.outputValues;
        outputLayer.gradients.push_back(matrix::mbe(outputLayerError, Utils::tanhOutputToDerivative(outputLayer.outputValues)));

        error = 0.0;
        for (int n = 0; n < outputLayer.outputValues.numColumns; n++) {
            double delta = y[0][n] - outputLayer.outputValues[n][0];
            error += delta * delta;
        }
        error /= outputLayer.outputValues.numColumns;
        error = (double) sqrt(error);

        recentAverageError =
                (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

        predicted[i] = outputLayer.outputValues[0][0];
        outputValues = &outputLayer.outputValues;
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
        layer->v = layer->v * mu + layer->updateW * eta;
        layer->W += layer->v;
        layer->b += layer->updateB * eta;
        layer->updateW *= 0;
        layer->updateB *= 0;

        RNNLayer* rnnLayer;
        if ((rnnLayer = dynamic_cast<RNNLayer*>(*it)) != nullptr) {
            rnnLayer->memoryV = rnnLayer->memoryV * mu + rnnLayer->updateMemoryW * eta;
            rnnLayer->memoryW += rnnLayer->memoryV; // eta * rnnLayer->updateMemoryW;//
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
        type(HIDDEN), W(matrix::random::rand(in,out) - 0.5), deltaW(in,out), b(matrix::random::rand(1,out) - 0.5), deltaB(1,out),
        outputValues(1,in), updateW(W.numRows, W.numColumns), v(W.numRows, W.numColumns), updateB(b.numRows, b.numColumns) {
}

void Layer::feedForward(Model &model, Layer &prevLayer) {
    outputValues = Utils::tanhFunction(prevLayer.outputValues*prevLayer.W + 1.0*prevLayer.b);
}

// TODO: check this working with basic layer!
void Layer::backPropagate(Model &model, Layer &nextLayer, const unsigned int& step) {
    matrix l0 = outputValues;
    RNNLayer* rnnLayer;
    Layer* layer = &nextLayer;
    matrix outputGradient(0,0);
    if ((rnnLayer = dynamic_cast<RNNLayer*>(layer)) != nullptr) {
        outputGradient = rnnLayer->gradientMemoryW;
    } else {
        outputGradient = nextLayer.gradients[step];
    }
    // TODO: momentum
    //v = v*model.mu + l0.transposed()*outputGradient;
    updateW += model.eta * l0.transposed()*outputGradient;
    updateB += model.eta * matrix::mbe(b, outputGradient);
}

void Layer::backPropagate(Model &model, Layer &nextLayer, matrix &XSet, const unsigned int &step) {
    // Output Values on input layer
    matrix ov0 = XSet;
    RNNLayer* rnnLayer;
    Layer* layer = &nextLayer;
    matrix outputGradient(0,0);
    if ((rnnLayer = dynamic_cast<RNNLayer*>(layer)) != nullptr) {
        outputGradient = rnnLayer->gradientMemoryW;
    } else {
        outputGradient = nextLayer.gradients[step];
    }
    // TODO: momentum
    //v = v*model.mu + ov0.transposed()*outputGradient;
    updateW += model.eta * ov0.transposed()*outputGradient;
    updateB += model.eta * matrix::mbe(b, outputGradient);
}

RNNLayer::RNNLayer(const unsigned int in, const unsigned int out, int capacity) :
        Layer::Layer(in, out), type(RECURRENT), memoryW(matrix::random::rand(in, in) - 0.5), capacity(capacity), updateMemoryW(memoryW.numRows, memoryW.numColumns),
        memoryV(memoryW.numRows, memoryW.numColumns), gradientMemoryW(1,in) {
    prevOutputValues.push_back(matrix(1, in));
}

void RNNLayer::feedForward(Model& model, Layer &prevLayer) {
    outputValues = Utils::tanhFunction(prevLayer.outputValues*prevLayer.W + prevOutputValues.back()*memoryW + 1.0*prevLayer.b);
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
    // Output & Previous Output Values on hidden (RNN) layer
    matrix& ov = prevOutputValues[step + 1];
    matrix& prevov = prevOutputValues[step];
    // TODO: momentum
    //v = v * model.mu + ov.transposed() * outputGradient;
    updateW += model.eta * ov.transposed() * outputGradient;
    updateB += model.eta * matrix::mbe(b, outputGradient);
    matrix hiddenGradient = matrix::mbe((gradientMemoryW*memoryW.transposed() + outputGradient*W.transposed()), Utils::tanhOutputToDerivative(ov));
    //memoryV = memoryV * model.mu + prevov.transposed()*hiddenGradient;
    updateMemoryW += model.eta * prevov.transposed()*hiddenGradient;
    gradientMemoryW = hiddenGradient;

}

LSTMLayer::LSTMLayer(const unsigned int in, const unsigned int out, int capacity) :
        RNNLayer(in, out, capacity), type(LSTM),
        forgetW(matrix::random::rand(in, in) - 0.5), updateForgetW(forgetW.numRows, forgetW.numColumns), gradientForgetW(1,in),
        forgetB(matrix::random::rand(1,out) - 0.5), deltaForgetB(1,out), updateForgetB(forgetB.numRows, forgetB.numColumns) {

}

void LSTMLayer::feedForward(Model &model, Layer &prevLayer) {
    RNNLayer::feedForward(model, prevLayer);
}

void LSTMLayer::backPropagate(Model &model, Layer &nextLayer, const unsigned int &step) {
    RNNLayer::backPropagate(model, nextLayer, step);
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

