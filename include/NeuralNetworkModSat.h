#ifndef NEURALNETWORKMODSAT_H
#define NEURALNETWORKMODSAT_H

#include <string>
#include "reluka.h"
#include "pwl2limodsat.h"
#include "VariableManager.h"
#include "LinearPiece.h"

namespace reluka
{
class NeuralNetworkModSat
{
    public:
        NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                            const std::vector<unsigned>& inputNnOutputIndexes,
                            std::string onnxFileName);
        NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                            const std::vector<unsigned>& inputNnOutputIndexes,
                            std::string onnxFileName,
                            bool inputNormalizeOutput);
        NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                            std::string onnxFileName);
        NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                            std::string onnxFileName,
                            bool inputNormalizeOutput);
        size_t getInputDimension() { return neuralNetwork.front().at(0).size()-1; }
        size_t getOutputDimension() { return neuralNetwork.back().size(); }
        void representNNmodsat();
        void printNNmodsatFile(unsigned outIdx);
        std::map<unsigned,std::pair<double,double>> getOriginalOutputLim();
        void printNNmodsat(std::ofstream *propertyFile, std::vector<pwl2limodsat::Variable> nnOutputVariables);

    private:
        std::vector<std::string> liModSatFileName;
        NeuralNetworkData neuralNetwork;
        pwl2limodsat::VariableManager *vm;

        std::vector<unsigned> nnOutputIndexes;
        std::vector<lukaFormula::Formula> outputFormulaRep;
        lukaFormula::ModsatSet outputModsatRep;
        std::map<unsigned,std::pair<double,double>> originalOutputLim;

        bool normalizeOutput = false;
        bool NNmodsatRepresentation = false;

        size_t getNnOutputIndexesIdx(unsigned nnOutputIndex);
        pwl2limodsat::LPCoefNonNegative gcd(pwl2limodsat::LPCoefNonNegative a,
                                            pwl2limodsat::LPCoefNonNegative b);
        pwl2limodsat::LinearPieceCoefficient dec2frac(NodeCoefficient decValue);

        void net2limodsatRec(const std::vector<NodeCoefficient>& normalizingNumbers,
                             const std::vector<pwl2limodsat::Variable>& inputVariables,
                             size_t layerNum);
        void net2limodsat();
};
}

#endif // NEURALNETWORKMODSAT_H
