#ifndef ZHANGBOLCSKEIMODSAT_H
#define ZHANGBOLCSKEIMODSAT_H

#include <string>
#include "reluka.h"
#include "pwl2limodsat.h"
#include "VariableManager.h"
#include "LinearPiece.h"

namespace reluka
{
class ZhangBolcskeiModSat
{
    public:
        ZhangBolcskeiModSat(const NeuralNetworkData& inputNeuralNetwork,
                            const std::vector<unsigned>& inputNnOutputIndexes,
                            std::string onnxFileName);
        ZhangBolcskeiModSat(const NeuralNetworkData& inputNeuralNetwork,
                            std::string onnxFileName);
        size_t getInputDimension() { return neuralNetwork.front().at(0).size()-1; }
        size_t getOutputDimension() { return neuralNetwork.back().size(); }
        void representZBmodsat();
        void printZBmodsatFile(unsigned outIdx);

    private:
        std::vector<std::string> liModSatFileName;
        NeuralNetworkData neuralNetwork;
        pwl2limodsat::VariableManager *vm;

        std::vector<unsigned> nnOutputIndexes;
        std::vector<lukaFormula::Formula> outputFormulaRep;
        lukaFormula::ModsatSet outputModsatRep;

        bool ZBmodsatRepresentation = false;

        size_t getNnOutputIndexesIdx(unsigned nnOutputIndex);
        pwl2limodsat::LPCoefNonNegative gcd(pwl2limodsat::LPCoefNonNegative a,
                                            pwl2limodsat::LPCoefNonNegative b);
        pwl2limodsat::LinearPieceCoefficient dec2frac(NodeCoefficient decValue);

        void net2limodsatRec(const std::vector<unsigned>& stretchingNumbers,
                             const std::vector<pwl2limodsat::Variable>& inputVariables,
                             size_t layerNum);
        void net2limodsat();
};
}

#endif // ZHANGBOLCSKEIMODSAT_H
