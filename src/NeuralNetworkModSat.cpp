#include <cmath>
#include "NeuralNetworkModSat.h"

#define PRECISION 1000000

namespace reluka
{
NeuralNetworkModSat::NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                                         const std::vector<unsigned>& inputNnOutputIndexes,
                                         std::string onnxFileName) :
    neuralNetwork(inputNeuralNetwork),
    nnOutputIndexes(inputNnOutputIndexes)
{
    std::string generalLiModSatFileName;

    if ( onnxFileName.substr(onnxFileName.size()-5,5) == ".onnx" )
        generalLiModSatFileName = onnxFileName.substr(0,onnxFileName.size()-5);
    else
        generalLiModSatFileName = onnxFileName;

    if ( nnOutputIndexes.empty() )
        for ( size_t outIdx = 0; outIdx < neuralNetwork.back().size(); outIdx++ )
            nnOutputIndexes.push_back(outIdx);

    for ( size_t outIdx = 0; outIdx < nnOutputIndexes.size(); outIdx++ )
        liModSatFileName.push_back(generalLiModSatFileName + "_" + std::to_string(nnOutputIndexes.at(outIdx)) + ".limodsat");

    vm = new pwl2limodsat::VariableManager(neuralNetwork.at(0).at(0).size()-1);
}

NeuralNetworkModSat::NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                                         const std::vector<unsigned>& inputNnOutputIndexes,
                                         std::string onnxFileName,
                                         bool inputNormalizeOutput) :
    NeuralNetworkModSat(inputNeuralNetwork,
                        inputNnOutputIndexes,
                        onnxFileName)
{
    normalizeOutput = inputNormalizeOutput;
}

NeuralNetworkModSat::NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                                         std::string onnxFileName) :
    NeuralNetworkModSat(inputNeuralNetwork,
                        std::vector<unsigned>(),
                        onnxFileName) {}

NeuralNetworkModSat::NeuralNetworkModSat(const NeuralNetworkData& inputNeuralNetwork,
                                         std::string onnxFileName,
                                         bool inputNormalizeOutput) :
    NeuralNetworkModSat(inputNeuralNetwork,
                        onnxFileName)
{
    normalizeOutput = inputNormalizeOutput;
}

size_t NeuralNetworkModSat::getNnOutputIndexesIdx(unsigned nnOutputIndex)
{
    size_t outIdx = -1;

    for ( size_t i = 0; i < nnOutputIndexes.size(); i++ )
        if ( nnOutputIndexes.at(i) == (size_t) nnOutputIndex )
            outIdx = i;

    if ( outIdx == -1 )
        throw std::invalid_argument("Such output NNmodsat representation was not built.");

    return outIdx;
}

pwl2limodsat::LPCoefNonNegative NeuralNetworkModSat::gcd(pwl2limodsat::LPCoefNonNegative a,
                                                         pwl2limodsat::LPCoefNonNegative b)
{
    if (a == 0)
        return b;
    else if (b == 0)
        return a;

    if (a < b)
        return gcd(a, b % a);
    else
        return gcd(b, a % b);
}

pwl2limodsat::LinearPieceCoefficient NeuralNetworkModSat::dec2frac(NodeCoefficient decValue)
{
    pwl2limodsat::LPCoefNonNegative whole;
    bool negFactor = false;

    if ( decValue >= 0 )
        whole = floor(decValue);
    else
    {
        whole = abs(ceil(decValue));
        negFactor = true;
    }

    pwl2limodsat::LPCoefNonNegative decimals = ( (negFactor ? -1 : 1) * decValue - whole ) * PRECISION;
    decimals = decimals + whole * PRECISION;
    pwl2limodsat::LPCoefNonNegative factor = gcd(decimals, PRECISION);

    pwl2limodsat::LPCoefNonNegative denominator = PRECISION / factor;
    pwl2limodsat::LPCoefInteger numerator = ( decimals / factor ) * (negFactor ? -1 : 1);

    return pwl2limodsat::LinearPieceCoefficient(numerator, denominator);
}

void NeuralNetworkModSat::net2limodsatRec(const std::vector<NodeCoefficient>& normalizingNumbers,
                                          const std::vector<pwl2limodsat::Variable>& inputVariables,
                                          size_t layerNum)
{
    if ( layerNum + 1 == neuralNetwork.size() )
    {
        for ( size_t outIdx = 0; outIdx < nnOutputIndexes.size(); outIdx++ )
        {
            if ( normalizeOutput )
            {
                NodeCoefficient minimum = neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(0);
                NodeCoefficient maximum = minimum;
                for ( size_t i = 1; i < neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).size(); i++ )
                {
                    if ( neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(i) <= 0 )
                        minimum += neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(i);
                    else
                        maximum += neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(i);
                }

                neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(0) -= minimum;
                for ( NodeCoefficient nCoeff : neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)) )
                    nCoeff /= maximum - minimum;

                originalOutputLim[nnOutputIndexes.at(outIdx)] = std::pair<double,double>(minimum,maximum);
            }

            pwl2limodsat::LinearPieceData lpData;

            lpData.push_back(dec2frac(neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(0)));
            for ( size_t i = 1; i < neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).size(); i++ )
                lpData.push_back(dec2frac(neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)).at(i)*normalizingNumbers.at(i-1)));

            pwl2limodsat::LinearPiece linearFunction(lpData, inputVariables, vm);
            linearFunction.representModsat();
            outputFormulaRep.push_back(linearFunction.getRepresentativeFormula());
            lukaFormula::ModsatSet auxModsatSet = linearFunction.getModsatSet();
            outputModsatRep.insert(outputModsatRep.end(), auxModsatSet.begin(), auxModsatSet.end());
        }
    }
    else
    {
        std::vector<NodeCoefficient> newNormalizingNumbers;
        std::vector<pwl2limodsat::Variable> newInputVariables;

        for ( Node node : neuralNetwork.at(layerNum) )
        {
            pwl2limodsat::LinearPieceData lpData;

            for ( size_t i = 1; i < node.size(); i++ )
                node.at(i) *= normalizingNumbers.at(i-1);

            NodeCoefficient maximum = node.at(0);
            for ( size_t i = 1; i < node.size(); i++ )
                if ( node.at(i) > 0 )
                    maximum += node.at(i);

            if ( maximum > 1 )
            {
                newNormalizingNumbers.push_back(maximum);

                for ( NodeCoefficient nCoeff : node )
                    lpData.push_back(dec2frac(nCoeff/maximum));
            }
            else
            {
                newNormalizingNumbers.push_back(1);

                for ( NodeCoefficient nCoeff : node )
                    lpData.push_back(dec2frac(nCoeff));
            }

            pwl2limodsat::LinearPiece linearFunction(lpData, inputVariables, vm);
            linearFunction.representModsat();
            outputModsatRep.push_back(linearFunction.getRepresentativeFormula());
            newInputVariables.push_back(vm->newVariable());
            outputModsatRep.back().addEquivalence(lukaFormula::Formula(newInputVariables.back()));
            lukaFormula::ModsatSet auxModsatSet = linearFunction.getModsatSet();
            outputModsatRep.insert(outputModsatRep.end(), auxModsatSet.begin(), auxModsatSet.end());
        }

        net2limodsatRec(newNormalizingNumbers, newInputVariables, layerNum+1);
    }
}

void NeuralNetworkModSat::net2limodsat()
{
    std::vector<NodeCoefficient> normalizingNumbers(neuralNetwork.at(0).at(0).size()-1, 1);
    std::vector<pwl2limodsat::Variable> inputVariables;

    for ( size_t i = 1; i < neuralNetwork.at(0).at(0).size(); i++ )
        inputVariables.push_back(pwl2limodsat::Variable(i));

    net2limodsatRec(normalizingNumbers, inputVariables, 0);

    NNmodsatRepresentation = true;
}

void NeuralNetworkModSat::representNNmodsat()
{
    if ( !NNmodsatRepresentation )
        net2limodsat();
}

void NeuralNetworkModSat::printNNmodsatFile(unsigned nnOutputIdx)
{
    size_t outIdx = getNnOutputIndexesIdx(nnOutputIdx);

    std::ofstream liModSatFile(liModSatFileName.at(outIdx));

    if ( !NNmodsatRepresentation )
        net2limodsat();

    liModSatFile << "-= Formula phi =-" << std::endl;
    outputFormulaRep.at(outIdx).print(&liModSatFile);

    liModSatFile << std::endl << "-= MODSAT Set Phi =-" << std::endl;

    for ( lukaFormula::Formula form : outputModsatRep )
    {
        liModSatFile << "f:" << std::endl;
        form.print(&liModSatFile);
    }
}

std::map<unsigned,std::pair<double,double>> NeuralNetworkModSat::getOriginalOutputLim()
{
    if ( !NNmodsatRepresentation )
        net2limodsat();

    return originalOutputLim;
}

void NeuralNetworkModSat::printNNmodsat(std::ofstream *propertyFile, std::vector<pwl2limodsat::Variable> nnOutputVariables)
{
    for ( lukaFormula::Formula form : outputModsatRep )
    {
        *propertyFile << "f:" << std::endl;
        form.print(propertyFile);
    }

    for ( size_t i = 0; i < outputFormulaRep.size(); i++ )
    {
        *propertyFile << "f:" << std::endl;
        outputFormulaRep.at(i).addEquivalence(lukaFormula::Formula(nnOutputVariables.at(i)));
        outputFormulaRep.at(i).print(propertyFile);
    }
}
}
