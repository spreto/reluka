#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include "reluka.h"
#include "pwl2limodsat.h"

namespace reluka
{
class NeuralNetwork
{
    public:
        NeuralNetwork(const NeuralNetworkData& inputNet, std::string onnxFileName, bool multithreading);
        NeuralNetwork(const NeuralNetworkData& inputNet, std::string onnxFileName);
        pwl2limodsat::PiecewiseLinearFunctionData getPwlData();
        pwl2limodsat::BoundaryPrototypeCollection getBoundProtData();
        void printPwlFile();
        std::string getPwlFileName() { return pwlFileName; }

    private:
        std::string pwlFileName;
        enum ProcessingMode { Single, Multi };
        ProcessingMode processingMode;

        NeuralNetworkData net;

        pwl2limodsat::PiecewiseLinearFunctionData pwlData;
        pwl2limodsat::BoundaryPrototypeCollection boundProtData;

        bool pwlTranslation = false;

        void setProcessingMode(ProcessingMode mode) { processingMode = mode; }

        pwl2limodsat::LPCoefNonNegative gcd(pwl2limodsat::LPCoefNonNegative a,
                                            pwl2limodsat::LPCoefNonNegative b);
        pwl2limodsat::LinearPieceCoefficient dec2frac(NodeCoefficient decValue);

        BoundProtPosition boundProtPosition(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                            pwl2limodsat::BoundProtIndex bIdx);
        bool feasibleBounds(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                            const pwl2limodsat::BoundaryCollection& boundData);
        pwl2limodsat::BoundaryPrototypeCollection composeBoundProtData(const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                                                                       unsigned layerNum);
        void writeBoundProtData(pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                const pwl2limodsat::BoundaryPrototypeCollection& newBoundProtData);
        pwl2limodsat::BoundaryPrototypeCollection composeOutputValues(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                                                      const std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                                                                      const std::vector<BoundProtPosition>& boundProtPositions);
        void writePwlData(pwl2limodsat::PiecewiseLinearFunctionData& pwlData,
                          const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                          const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                          const pwl2limodsat::BoundaryCollection& currentBoundData,
                          pwl2limodsat::BoundProtIndex newBoundProtFirstIdx);
        bool iterate(size_t limitIterationIdx,
                     std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                     size_t& currentIterationIdx,
                     const std::vector<BoundProtPosition>& boundProtPositions,
                     pwl2limodsat::BoundaryCollection& boundData);
        bool iterate(std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                     size_t& currentIterationIdx,
                     const std::vector<BoundProtPosition>& boundProtPositions,
                     pwl2limodsat::BoundaryCollection& boundData);

        void net2pwl(pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                     pwl2limodsat::PiecewiseLinearFunctionData& pwlData,
                     const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                     const pwl2limodsat::BoundaryCollection& currentBoundData,
                     size_t layerNum);
        void net2pwl(const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                     const pwl2limodsat::BoundaryCollection& currentBoundData,
                     size_t layerNum);

        std::pair<pwl2limodsat::PiecewiseLinearFunctionData,
                  pwl2limodsat::BoundaryPrototypeCollection> partialNet2pwl(unsigned startingPoint,
                                                                            size_t fixedNodesNum,
                                                                            const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                                                                            const std::vector<BoundProtPosition>& boundProtPositions);
        void pwlInfoMerge(const std::vector<std::pair<pwl2limodsat::PiecewiseLinearFunctionData,
                                                      pwl2limodsat::BoundaryPrototypeCollection>>& threadsInfo);
        void net2pwlMultithreading(const pwl2limodsat::BoundaryPrototypeCollection& inputValues);

        void net2pwl();
};
}

#endif // NEURALNETWORK_H
