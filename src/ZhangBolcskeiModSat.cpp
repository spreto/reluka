#include <cmath>
#include "ZhangBolcskeiModSat.h"

#define PRECISION 1000000

namespace reluka
{
ZhangBolcskeiModSat::ZhangBolcskeiModSat(const NeuralNetworkData& inputNeuralNetwork,
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

ZhangBolcskeiModSat::ZhangBolcskeiModSat(const NeuralNetworkData& inputNeuralNetwork,
                                         std::string onnxFileName) :
    ZhangBolcskeiModSat(inputNeuralNetwork,
                        std::vector<unsigned>(),
                        onnxFileName) {}

size_t ZhangBolcskeiModSat::getNnOutputIndexesIdx(unsigned nnOutputIndex)
{
    size_t outIdx = -1;

    for ( size_t i = 0; i < nnOutputIndexes.size(); i++ )
        if ( nnOutputIndexes.at(i) == (size_t) nnOutputIndex )
            outIdx = i;

    if ( outIdx == -1 )
        throw std::invalid_argument("Such output ZBmodsat representation was not built.");

    return outIdx;
}

pwl2limodsat::LPCoefNonNegative ZhangBolcskeiModSat::gcd(pwl2limodsat::LPCoefNonNegative a,
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

pwl2limodsat::LinearPieceCoefficient ZhangBolcskeiModSat::dec2frac(NodeCoefficient decValue)
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

void ZhangBolcskeiModSat::net2limodsatRec(const std::vector<unsigned>& stretchingNumbers,
                                          const std::vector<pwl2limodsat::Variable>& inputVariables,
                                          size_t layerNum)
{
    if ( layerNum + 1 == neuralNetwork.size() )
    {
        for ( size_t outIdx = 0; outIdx < nnOutputIndexes.size(); outIdx++ )
        {
            pwl2limodsat::LinearPieceData lpData;
            for ( NodeCoefficient nCoeff : neuralNetwork.at(layerNum).at(nnOutputIndexes.at(outIdx)) )
                lpData.push_back(dec2frac(nCoeff));

            size_t i = 1;
            for ( unsigned stretch : stretchingNumbers )
            {
                lpData.insert(lpData.begin()+i, stretch-1, lpData.at(i));
                i += stretch;
            }

            pwl2limodsat::LinearPiece linearFunction(lpData, inputVariables, vm);
            linearFunction.representModsat();
            outputFormulaRep.push_back(linearFunction.getRepresentativeFormula());
            lukaFormula::ModsatSet auxModsatSet = linearFunction.getModsatSet();
            outputModsatRep.insert(outputModsatRep.end(), auxModsatSet.begin(), auxModsatSet.end());
        }
    }
    else
    {
        std::vector<unsigned> newStretchingNumbers;
        std::vector<pwl2limodsat::Variable> newInputVariables;

        for ( Node node : neuralNetwork.at(layerNum) )
        {
            pwl2limodsat::LinearPieceData lpData;
            for ( NodeCoefficient nCoeff : node )
                lpData.push_back(dec2frac(nCoeff));

            size_t i = 1;
            for ( unsigned stretch : stretchingNumbers )
            {
                lpData.insert(lpData.begin()+i, stretch-1, lpData.at(i));
                i += stretch;
            }

            NodeCoefficient maximum = 0;
            for ( NodeCoefficient nCoeff : node )
                if ( nCoeff >= 0 )
                    maximum += nCoeff;

            if ( ceil(maximum) == 0 )
            {
                newStretchingNumbers.push_back(1);
                outputModsatRep.push_back(pwl2limodsat::LinearPiece::zeroFormula(vm));
                newInputVariables.push_back(vm->newVariable());
                outputModsatRep.back().addEquivalence(lukaFormula::Formula(newInputVariables.back()));
            }
            else
            {
                newStretchingNumbers.push_back(ceil(maximum));

                for ( i = 0; i < ceil(maximum); i++ )
                {
                    lpData.at(0).first -= ( i ? lpData.at(0).second : 0 );
                    pwl2limodsat::LinearPiece linearFunction(lpData, inputVariables, vm);
                    linearFunction.representModsat();
                    outputModsatRep.push_back(linearFunction.getRepresentativeFormula());
                    newInputVariables.push_back(vm->newVariable());
                    outputModsatRep.back().addEquivalence(lukaFormula::Formula(newInputVariables.back()));
                    lukaFormula::ModsatSet auxModsatSet = linearFunction.getModsatSet();
                    outputModsatRep.insert(outputModsatRep.end(), auxModsatSet.begin(), auxModsatSet.end());
                }
            }
        }

        net2limodsatRec(newStretchingNumbers, newInputVariables, layerNum+1);
    }
}

void ZhangBolcskeiModSat::net2limodsat()
{
    std::vector<unsigned> stretchingNumbers(neuralNetwork.at(0).at(0).size()-1, 1);
    std::vector<pwl2limodsat::Variable> inputVariables;

    for ( size_t i = 1; i < neuralNetwork.at(0).at(0).size(); i++ )
        inputVariables.push_back(pwl2limodsat::Variable(i));

    net2limodsatRec(stretchingNumbers, inputVariables, 0);

    ZBmodsatRepresentation = true;
}

void ZhangBolcskeiModSat::representZBmodsat()
{
    if ( !ZBmodsatRepresentation )
        net2limodsat();
}

void ZhangBolcskeiModSat::printZBmodsatFile(unsigned nnOutputIdx)
{
    size_t outIdx = getNnOutputIndexesIdx(nnOutputIdx);

    std::ofstream liModSatFile(liModSatFileName.at(outIdx));

    if ( !ZBmodsatRepresentation )
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
}
