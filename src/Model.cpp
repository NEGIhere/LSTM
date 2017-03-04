//
// Created by negig on 23.02.2017.
//

#include <assert.h>
#include <algorithm>
#include "Model.h"

Model::Model(double learningRate, double momentum, double dropout) : dropout(dropout) {
    eta = learningRate;
    mu = (1.0 - momentum);

    outputValues = nullptr;
}

void Model::addLayer(Layer* layer) {
    /*if (layers.size() == 0 && layer->type != INPUT) {
        printf("First layer type must be INPUT");
        exit(1);
    }*/
    layers.push_back(layer);
}

void Model::train(std::vector<matrix>& XSet, std::vector<matrix>& ySet, double* predicted) {
    const unsigned int size = (unsigned int)XSet.size();
    assert(size == ySet.size());
    setSize = size;
    Layer &outputLayer = *layers.back();

    if (Utils::randDouble(0.0, 1.0) <= dropout) {
        layers[Utils::randInt(0, layers.size() - 1)]->dropout();
    }

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
                currLayer.backPropagate(*this, (unsigned int) i, nextLayer, XSet[size - step - 1], size - step - 1); // size - step - 1
            } else {
                currLayer.backPropagate(*this, (unsigned int) i, nextLayer, size - step - 1);
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

        if (layer->type == RECURRENT) {
            RNNLayer *rnnLayer;
            if ((rnnLayer = dynamic_cast<RNNLayer *>(*it)) != nullptr) {
                rnnLayer->memoryV = rnnLayer->memoryV * mu + rnnLayer->updateMemoryW * eta;
                rnnLayer->memoryW += rnnLayer->memoryV; // eta * rnnLayer->updateMemoryW;//
                rnnLayer->updateMemoryW *= 0;

                rnnLayer->prevOutputValues.clear();
                rnnLayer->prevOutputValues.push_back(
                        matrix(rnnLayer->outputValues.numRows, rnnLayer->outputValues.numColumns));
                rnnLayer->outputValues *= 0;
                rnnLayer->gradientMemoryW *= 0;
            }
        } else if (layer->type == LSTM) {
            LSTMLayer *lstmLayer = dynamic_cast<LSTMLayer *>(*it);

            //lstmLayer->memoryV = lstmLayer->memoryV * mu + lstmLayer->updateMemoryW * eta;
            //lstmLayer->memoryW += lstmLayer->memoryV; // eta * rnnLayer->updateMemoryW;//

            lstmLayer->candidateCellStateW += eta * lstmLayer->updateCandidateCellStateW;
            lstmLayer->inputGateW += eta * lstmLayer->updateInputGateW;
            lstmLayer->forgetW += eta * lstmLayer->updateForgetW;
            lstmLayer->sigmoidGateW += eta * lstmLayer->updateSigmoidGateW;
            lstmLayer->candidateCellStateB += eta * lstmLayer->updateCandidateCellStateB;
            lstmLayer->inputGateB += eta * lstmLayer->updateInputGateB;
            lstmLayer->forgetB += eta * lstmLayer->updateForgetB;
            lstmLayer->sigmoidGateB += eta * lstmLayer->updateSigmoidGateB;

            lstmLayer->updateCandidateCellStateW *= 0;
            lstmLayer->updateInputGateW *= 0;
            lstmLayer->updateForgetW *= 0;
            lstmLayer->updateSigmoidGateW *= 0;
            lstmLayer->updateCandidateCellStateB *= 0;
            lstmLayer->updateInputGateB *= 0;
            lstmLayer->updateForgetB *= 0;
            lstmLayer->updateSigmoidGateB *= 0;
            lstmLayer->updateMemoryW *= 0;

            lstmLayer->prevOutputValues.clear();
            lstmLayer->prevOutputValues.push_back(matrix(lstmLayer->outputValues.numRows, lstmLayer->outputValues.numColumns));
            lstmLayer->outputValues *= 0;
            lstmLayer->gradientMemoryW *= 0;
        }
        (*it)->gradients.clear();
    }
}

matrix Model::getOutputValues() {
    return *outputValues;
}

Layer::Layer(const unsigned int in, const unsigned int out) :
        W(matrix::random::rand(in,out) - 0.5), deltaW(in,out), b(matrix::random::rand(1,out) - 0.5), deltaB(1,out),
        outputValues(1,in), updateW(W.numRows, W.numColumns), v(W.numRows, W.numColumns), updateB(b.numRows, b.numColumns) {
    type = HIDDEN;
}

void Layer::dropout() {
    matrix* w;
    if (Utils::randInt(0,1) == 0) {
        w = &W;
    } else {
        w = &b;
    }
    (*w)[Utils::randInt(0, w->numRows - 1)][Utils::randInt(0, w->numColumns - 1)] = 0.0;
}

void Layer::feedForward(Model &model, Layer &prevLayer) {
    outputValues = Utils::tanh(prevLayer.outputValues * prevLayer.W + 1.0 * prevLayer.b);
}

// TODO: check this working with basic layer!
void Layer::backPropagate(Model &model, const unsigned int& li, Layer &nextLayer, const unsigned int& step) {
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
    updateW += l0.transposed()*outputGradient;
    updateB += matrix::mbe(b, outputGradient);
}

void Layer::backPropagate(Model &model, const unsigned int& li, Layer &nextLayer, matrix &X, const unsigned int &step) {
    // Output Values on input layer
    matrix ov0 = X;
    Layer* nLayer = &nextLayer;
    matrix* outputGradient;
    matrix delta(0,0);

    if (nextLayer.type == LSTM) {
        LSTMLayer* lstmLayer = dynamic_cast<LSTMLayer*>(nLayer);
        //nextLayer.gradients[model.setSize - 1 - step][0][0] += ;

        matrix dC = matrix::mbe(lstmLayer->sigmoidGate, nextLayer.gradients[model.setSize - 1 - step] + lstmLayer->diffH) + lstmLayer->diffCellState;
        matrix dSigmoid = matrix::mbe(lstmLayer->cellState, nextLayer.gradients[model.setSize - 1 - step] + lstmLayer->diffH);
        matrix di = matrix::mbe(lstmLayer->candidateCellState, dC);
        matrix dg = matrix::mbe(lstmLayer->inputGate, dC);
        matrix df = matrix::mbe(lstmLayer->prevCellState, dC);

        matrix diInput = matrix::mbe(di, Utils::sigmoidOutputToDerivative(lstmLayer->inputGate));
        matrix dfInput = matrix::mbe(df, Utils::sigmoidOutputToDerivative(lstmLayer->forget));
        matrix dSigmoidInput = matrix::mbe(dSigmoid, Utils::sigmoidOutputToDerivative(lstmLayer->sigmoidGate));
        matrix dgInput = matrix::mbe(dg, Utils::tanhOutputToDerivative(lstmLayer->candidateCellState));

        lstmLayer->updateInputGateW += Utils::outer(lstmLayer->xc[0], diInput[0]);
        lstmLayer->updateForgetW += Utils::outer(lstmLayer->xc[0], dfInput[0]);
        lstmLayer->updateSigmoidGateW += Utils::outer(lstmLayer->xc[0], dSigmoidInput[0]);
        lstmLayer->updateCandidateCellStateW += Utils::outer(lstmLayer->xc[0], dgInput[0]);
        lstmLayer->updateInputGateB += diInput;
        lstmLayer->updateForgetB += dfInput;
        lstmLayer->updateSigmoidGateB += dSigmoidInput;
        lstmLayer->updateCandidateCellStateB += dgInput;

        /*
        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dxc[:self.param.x_dim] # unused
        self.state.bottom_diff_h = dxc[self.param.x_dim:]
         */
        matrix dxc(lstmLayer->xc.numRows, lstmLayer->xc.numColumns);
        dxc += diInput*lstmLayer->inputGateW.transposed();
        dxc += dfInput*lstmLayer->forgetW.transposed();
        dxc += dSigmoidInput*lstmLayer->sigmoidGateW.transposed();
        dxc += dgInput*lstmLayer->candidateCellStateW.transposed();

        lstmLayer->diffCellState = matrix::mbe(dC, lstmLayer->forget);
        std::vector<double> v = dxc[0];
        v.erase(v.begin(), v.size() > 2 ? v.begin() + 2 : v.end()); // FIXME: xDim = n
        lstmLayer->diffH[0] = v;
    } else if (nextLayer.type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        delta = matrix::mbe(rnnLayer->gradientMemoryW*rnnLayer->memoryW.transposed() + nextLayer.gradients[model.setSize - 1 - step]*rnnLayer->W.transposed(), Utils::tanhOutputToDerivative(rnnLayer->prevOutputValues[step+1]));
        rnnLayer->gradientMemoryW = delta;
    } else {
        delta = nextLayer.gradients[step];
    }

    if (nextLayer.type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        rnnLayer->updateMemoryW += rnnLayer->prevOutputValues[step].transposed()*delta;
    } else if (nextLayer.type == LSTM) {
        LSTMLayer* lstmLayer = dynamic_cast<LSTMLayer*>(nLayer);
        //lstmLayer->updateMemoryW += lstmLayer->prevOutputValues[step].transposed()*delta;
    }
    if (nextLayer.type != LSTM) {
        updateW += ov0.transposed() * delta;
        updateB += matrix::mbe(b, delta);
    }
    gradients.push_back(delta);
}

RNNLayer::RNNLayer(const unsigned int in, const unsigned int out, int capacity) :
        Layer::Layer(in, out), /*type(RECURRENT), */memoryW(matrix::random::rand(in, in) - 0.5), capacity(capacity), updateMemoryW(memoryW.numRows, memoryW.numColumns),
        memoryV(memoryW.numRows, memoryW.numColumns), gradientMemoryW(1,in) {
    prevOutputValues.push_back(matrix(1, in));
    type = RECURRENT;
}

void RNNLayer::feedForward(Model& model, Layer &prevLayer) {
    outputValues = Utils::tanh(prevLayer.outputValues * prevLayer.W + prevOutputValues.back() * memoryW + 1.0 * prevLayer.b);
    prevOutputValues.push_back(outputValues);
    model.recurrentSteps++;
}

void RNNLayer::backPropagate(Model &model, const unsigned int& li, Layer &nextLayer, const unsigned int& step) {
    Layer* nLayer = &nextLayer;

    matrix* cellStateGradient = nullptr;
    matrix delta(0,0);

    if (nLayer->type == LSTM) {
        LSTMLayer* lstmLayer = dynamic_cast<LSTMLayer*>(nLayer);
        delta = lstmLayer->gradientMemoryW;
        cellStateGradient = &lstmLayer->gradientCellStateW;
    } else if (nLayer->type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        delta = matrix::mbe((rnnLayer->gradientMemoryW*rnnLayer->memoryW.transposed() + nextLayer.gradients[model.setSize - 1 - step]*rnnLayer->W.transposed()), Utils::tanhOutputToDerivative(rnnLayer->prevOutputValues[step+1]));
        gradients.push_back(delta);
        rnnLayer->gradientMemoryW = delta;
    } else {
        delta = nextLayer.gradients[step];
        gradients.push_back(delta);

    }
    matrix &ov = prevOutputValues[step + 1];

    if (nextLayer.type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        rnnLayer->updateMemoryW += rnnLayer->prevOutputValues[step].transposed()*delta;
    }
    // TODO: momentum
    //v = v * model.mu + ov.transposed() * outputGradient;
    updateW += ov.transposed() * delta;
    updateB += matrix::mbe(b, delta);
}

LSTMLayer::LSTMLayer(const unsigned int prevIn, const unsigned int in, const unsigned int out, int capacity) :
        RNNLayer(in, out, capacity),

        forget(1,in), forgetW(matrix::random::rand(in+prevIn,in) - 0.5), updateForgetW(forgetW.numRows, forgetW.numColumns), gradientForgetW(1,in),
        forgetB(matrix::random::rand(1,in) - 0.5), deltaForgetB(1,out), updateForgetB(forgetB.numRows, forgetB.numColumns),

        inputGate(1,in), inputGateW(matrix::random::rand(in+prevIn,in) - 0.5), updateInputGateW(inputGateW.numRows, inputGateW.numColumns), gradientInputGateW(1,in),
        inputGateB(matrix::random::rand(1,in) - 0.5), deltaInputGateB(1,out), updateInputGateB(inputGateB.numRows, inputGateB.numColumns),

        cellStateW(matrix::random::rand(in+prevIn,in) - 0.5), updateCellStateW(cellStateW.numRows, cellStateW.numColumns), gradientCellStateW(1,in),
        cellStateB(matrix::random::rand(1,in) - 0.5), deltaCellStateB(1,out), updateCellStateB(cellStateB.numRows, cellStateB.numColumns), cellState(1,in), prevCellState(cellState.numRows, cellState.numColumns),

        candidateCellStateW(matrix::random::rand(in+prevIn,in) - 0.5), updateCandidateCellStateW(candidateCellStateW.numRows, candidateCellStateW.numColumns), gradientCandidateCellStateW(1,in),
        candidateCellStateB(matrix::random::rand(1,in) - 0.5), deltaCandidateCellStateB(1,out), updateCandidateCellStateB(candidateCellStateB.numRows, candidateCellStateB.numColumns), candidateCellState(1,in),

        sigmoidGate(1,in), sigmoidGateW(matrix::random::rand(in+prevIn,in) - 0.5), updateSigmoidGateW(sigmoidGateW.numRows, sigmoidGateW.numColumns), gradientSigmoidGateW(1,in),
        sigmoidGateB(matrix::random::rand(1,in) - 0.5), deltaSigmoidGateB(1,out), updateSigmoidGateB(sigmoidGateB.numRows, sigmoidGateB.numColumns),
        xc(in+prevIn,in), diffCellState(1,in), diffH(1,in) {
    type = LSTM;
    //prevOutputValues.push_back(matrix(1, in));
}

void LSTMLayer::feedForward(Model &model, Layer &prevLayer) {
//    RNNLayer::feedForward(model, prevLayer);
//    outputValues = Utils::tanhFunction(prevLayer.outputValues*prevLayer.W + prevOutputValues.back()*memoryW + 1.0*prevLayer.b);
//    prevOutputValues.push_back(outputValues);
    xc = matrix::hstack(prevLayer.outputValues, prevOutputValues.back());
    forget = Utils::sigmoid(xc*forgetW + forgetB);
    inputGate = Utils::sigmoid(xc*inputGateW + inputGateB);
    candidateCellState = Utils::tanh(xc * candidateCellStateW + candidateCellStateB);

    prevCellState = cellState;
    cellState = matrix::mbe(prevCellState, forget) + matrix::mbe(candidateCellState, inputGate);
    sigmoidGate = Utils::sigmoid(xc*sigmoidGateW + sigmoidGateB);
    outputValues = matrix::mbe(Utils::tanh(cellState), sigmoidGate);
    prevOutputValues.push_back(outputValues);

    diffH *= 0;
    diffCellState *= 0;
}

void LSTMLayer::backPropagate(Model &model, const unsigned int& li, Layer &nextLayer, const unsigned int &step) {
    Layer* nLayer = &nextLayer;

    matrix* cellStateGradient = nullptr;
    matrix delta(0,0);

    if (nLayer->type == LSTM) {
        LSTMLayer* lstmLayer = dynamic_cast<LSTMLayer*>(nLayer);
        delta = lstmLayer->gradientMemoryW;
        cellStateGradient = &lstmLayer->gradientCellStateW;
    } else if (nLayer->type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        delta = matrix::mbe((rnnLayer->gradientMemoryW*rnnLayer->memoryW.transposed() + nextLayer.gradients[model.setSize - 1 - step]*rnnLayer->W.transposed()), Utils::tanhOutputToDerivative(rnnLayer->prevOutputValues[step+1]));
        gradients.push_back(delta);
        rnnLayer->gradientMemoryW = delta;
    } else {
        delta = nextLayer.gradients[step];
        // FIXME: PLEASE
        matrix g(1,16);
        g[0][0] = delta[0][0];
        gradients.push_back(g);

    }
    matrix &ov = prevOutputValues[step + 1];

    if (nextLayer.type == RECURRENT) {
        RNNLayer* rnnLayer = dynamic_cast<RNNLayer*>(nLayer);
        rnnLayer->updateMemoryW += rnnLayer->prevOutputValues[step].transposed()*delta;
    }
    // TODO: momentum
    updateW += ov.transposed() * delta;
    updateB += matrix::mbe(b, delta);
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

