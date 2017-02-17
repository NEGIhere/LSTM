
#ifndef NNTEST_NET_H
#define NNTEST_NET_H

#include "Neuron.h"

enum MemoryConnectionType {
    MANY_TO_MANY,
    ONE_TO_ONE
};

class Net {
public:
    Net(const std::vector<unsigned int> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void backPropThroughTime(const std::vector<double> &targetVals);
    void backPropThroughTimeOutput(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return recentAverageError; }
    double getError(void) const { return error; }
    void getWeights(std::vector<double> &weightVals, bool bias);
    void setWeights(const std::vector<double> &weightVals, bool bias);
    int getConnectionsCount(bool bias) const;
    void clearMemory();

    const std::vector<Layer> &getLayers() const;

    bool isFirstRunning() const;
    static const MemoryConnectionType memoryConnectionType = ONE_TO_ONE;

    virtual ~Net();
private:
    std::vector<Layer> layers;
    double error, recentAverageError;
    static double recentAverageSmoothingFactor;
    bool firstRunning;
};

#endif //NNTEST_NET_H
