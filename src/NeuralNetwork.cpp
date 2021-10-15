#include <fstream>
#include <cmath>
#include <future>
#include "soplex.h"
#include "NeuralNetwork.h"

#define PRECISION 1000000

namespace reluka
{
NeuralNetwork::NeuralNetwork(const NeuralNetworkData& inputNeuralNetwork,
                             const std::vector<unsigned>& inputNnOutputIndexes,
                             std::string onnxFileName,
                             bool multithreading) :
    neuralNetwork(inputNeuralNetwork),
    nnOutputIndexes(inputNnOutputIndexes)
{
    std::string generalPwlFileName;

    if ( onnxFileName.substr(onnxFileName.size()-5,5) == ".onnx" )
        generalPwlFileName = onnxFileName.substr(0,onnxFileName.size()-5);
    else
        generalPwlFileName = onnxFileName;

    if ( nnOutputIndexes.empty() )
        for ( size_t outIdx = 0; outIdx < neuralNetwork.back().size(); outIdx++ )
            nnOutputIndexes.push_back(outIdx);

    for ( size_t outIdx = 0; outIdx < nnOutputIndexes.size(); outIdx++ )
    {
        pwl2limodsat::PiecewiseLinearFunctionData emptyPwlData;
        pwlData.push_back(emptyPwlData);

        pwlFileName.push_back(generalPwlFileName + "_" + std::to_string(nnOutputIndexes.at(outIdx)) + ".pwl");
    }

    processingMode = ( multithreading ? Multi : Single );
}

NeuralNetwork::NeuralNetwork(const NeuralNetworkData& inputNeuralNetwork,
                             std::string onnxFileName,
                             bool multithreading) :
    NeuralNetwork(inputNeuralNetwork,
                  std::vector<unsigned>(),
                  onnxFileName,
                  multithreading) {}

NeuralNetwork::NeuralNetwork(const NeuralNetworkData& inputNeuralNetwork,
                             const std::vector<unsigned>& inputNnOutputIndexes,
                             std::string onnxFileName) :
    NeuralNetwork(inputNeuralNetwork, inputNnOutputIndexes, onnxFileName, true) {}

NeuralNetwork::NeuralNetwork(const NeuralNetworkData& inputNeuralNetwork,
                             std::string onnxFileName) :
    NeuralNetwork(inputNeuralNetwork, onnxFileName, true) {}

size_t NeuralNetwork::getNnOutputIndexesIdx(unsigned nnOutputIndex)
{
    size_t outIdx = -1;

    for ( size_t i = 0; i < nnOutputIndexes.size(); i++ )
        if ( nnOutputIndexes.at(i) == (size_t) nnOutputIndex )
            outIdx = i;

    if ( outIdx == -1 )
        throw std::invalid_argument("Such output pwl representation was not built.");

    return outIdx;
}

pwl2limodsat::LPCoefNonNegative NeuralNetwork::gcd(pwl2limodsat::LPCoefNonNegative a,
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

template<class T>
pwl2limodsat::LinearPieceCoefficient NeuralNetwork::dec2frac(T decValue)
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

BoundProtPosition NeuralNetwork::boundProtPosition(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                                   pwl2limodsat::BoundProtIndex bIdx)
{
    soplex::SoPlex sop;

    pwl2limodsat::BoundaryCoefficient K = -boundProtData.at(bIdx).at(0);

    soplex::DSVector dummycol(0);
    for ( size_t i = 1; i < boundProtData.at(bIdx).size(); i++ )
        sop.addColReal(soplex::LPCol(boundProtData.at(bIdx).at(i), dummycol, 1, 0));

    sop.setIntParam(soplex::SoPlex::VERBOSITY, soplex::SoPlex::VERBOSITY_ERROR);
    sop.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MAXIMIZE);
    sop.optimize();
    float Max = sop.objValueReal();

    sop.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    sop.optimize();
    float Min = sop.objValueReal();

    if ( (Min >= K) && (Max >= K) )
        return Over;
    else if ( (Max <= K) && (Min <= K) )
        return Under;
    else
        return Cutting;
}

bool NeuralNetwork::feasibleBounds(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                   const pwl2limodsat::BoundaryCollection& boundData)
{
    soplex::SoPlex sop;

    soplex::DSVector dummycol(0);
    for ( size_t i = 1; i <= neuralNetwork.at(0).at(0).size()-1; i++ )
        sop.addColReal(soplex::LPCol(0, dummycol, 1, 0));

    soplex::DSVector row(neuralNetwork.at(0).size());
    for ( size_t i = 0; i < boundData.size(); i++ )
    {
        for ( size_t j = 1; j <= neuralNetwork.at(0).at(0).size()-1; j++ )
            row.add(j-1, boundProtData.at(boundData.at(i).first).at(j));

        if ( boundData.at(i).second == pwl2limodsat::GeqZero )
            sop.addRowReal(soplex::LPRow(-boundProtData.at(boundData.at(i).first).at(0), row, soplex::infinity));
        else if ( boundData.at(i).second == pwl2limodsat::LeqZero )
            sop.addRowReal(soplex::LPRow(-soplex::infinity, row, -boundProtData.at(boundData.at(i).first).at(0)));

        row.clear();
    }

    sop.setIntParam(soplex::SoPlex::VERBOSITY, soplex::SoPlex::VERBOSITY_ERROR);
    sop.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MAXIMIZE);
    sop.optimize();
    float Max = sop.objValueReal();

    if ( Max < 0 )
        return false;
    else
        return true;
}

pwl2limodsat::BoundaryPrototypeCollection NeuralNetwork::composeBoundProtData(const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                                                                              unsigned layerNum)
{
    pwl2limodsat::BoundaryPrototypeCollection newBoundProtData;

    for ( size_t i = 0; i < neuralNetwork.at(layerNum).size(); i++ )
    {
        pwl2limodsat::BoundaryPrototype auxBoundProt;

        for ( size_t j = 0; j < inputValues.at(0).size(); j++ )
        {
            pwl2limodsat::BoundaryCoefficient auxCoeff;

            if ( j == 0 )
                auxCoeff = neuralNetwork.at(layerNum).at(i).at(0);
            else
                auxCoeff = 0;

            for ( size_t k = 0; k < inputValues.size(); k++ )
                auxCoeff = auxCoeff + inputValues.at(k).at(j) * neuralNetwork.at(layerNum).at(i).at(k+1);

            auxBoundProt.push_back(auxCoeff);
        }

        newBoundProtData.push_back(auxBoundProt);
    }

    return newBoundProtData;
}

void NeuralNetwork::writeBoundProtData(pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                       const pwl2limodsat::BoundaryPrototypeCollection& newBoundProtData)
{
    for ( size_t i = 0; i < newBoundProtData.size(); i++ )
    {
        pwl2limodsat::BoundaryPrototype auxBoundProt;

        for ( size_t j = 0; j < newBoundProtData.at(0).size(); j++ )
        {
            pwl2limodsat::BoundaryCoefficient auxCoeff;
            auxCoeff = newBoundProtData.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProtData.push_back(auxBoundProt);
    }
}

pwl2limodsat::BoundaryPrototypeCollection NeuralNetwork::composeOutputValues(const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                                                             const std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                                                                             const std::vector<BoundProtPosition>& boundProtPositions)
{
    pwl2limodsat::BoundaryPrototypeCollection outputValues = boundProtData;

    for ( size_t j = 0; j < iteration.size(); j++ )
    {
        if ( boundProtPositions.at(j) == Cutting )
        {
            if ( iteration.at(j) == pwl2limodsat::LeqZero )
                for ( size_t k = 0; k < outputValues.at(0).size(); k++ )
                    outputValues.at(j).at(k) = 0;
        }
        else if ( boundProtPositions.at(j) == Under )
        {
            for ( size_t k = 0; k < outputValues.at(0).size(); k++ )
                outputValues.at(j).at(k) = 0;
        }
    }

    return outputValues;
}

void NeuralNetwork::writePwlData(pwl2limodsat::PiecewiseLinearFunctionData& pwlData,
                                 const pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                                 const pwl2limodsat::BoundaryPrototype& inputValues,
                                 const pwl2limodsat::BoundaryCollection& currentBoundData,
                                 std::pair<pwl2limodsat::BoundProtIndex,pwl2limodsat::BoundProtIndex> newBoundProtIdx)
{
    pwl2limodsat::RegionalLinearPieceData rlp0;
    rlp0.bound = currentBoundData;
    rlp0.bound.push_back(pwl2limodsat::Boundary(newBoundProtIdx.first, pwl2limodsat::LeqZero));
    if ( feasibleBounds(boundProtData, rlp0.bound) )
    {
        for ( size_t i = 0; i < inputValues.size(); i++ )
            rlp0.lpData.push_back(pwl2limodsat::LinearPieceCoefficient(0,1));
        pwlData.push_back(rlp0);
    }

    pwl2limodsat::RegionalLinearPieceData rlp0_1;
    rlp0_1.bound = currentBoundData;
    rlp0_1.bound.push_back(pwl2limodsat::Boundary(newBoundProtIdx.first, pwl2limodsat::GeqZero));
    rlp0_1.bound.push_back(pwl2limodsat::Boundary(newBoundProtIdx.second, pwl2limodsat::LeqZero));
    if ( feasibleBounds(boundProtData, rlp0_1.bound) )
    {
        for ( size_t i = 0; i < inputValues.size(); i++ )
            rlp0_1.lpData.push_back(dec2frac(inputValues.at(i)));
        pwlData.push_back(rlp0_1);
    }

    pwl2limodsat::RegionalLinearPieceData rlp1;
    rlp1.bound = currentBoundData;
    rlp1.bound.push_back(pwl2limodsat::Boundary(newBoundProtIdx.second, pwl2limodsat::GeqZero));

    if ( feasibleBounds(boundProtData, rlp1.bound) )
    {
        rlp1.lpData.push_back(pwl2limodsat::LinearPieceCoefficient(1,1));
        for ( size_t i = 1; i < inputValues.size(); i++ )
            rlp1.lpData.push_back(pwl2limodsat::LinearPieceCoefficient(0,1));
        pwlData.push_back(rlp1);
    }
}

bool NeuralNetwork::iterate(size_t limitIterationIdx,
                            std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                            size_t& currentIterationIdx,
                            const std::vector<BoundProtPosition>& boundProtPositions,
                            pwl2limodsat::BoundaryCollection& boundData)
{
    bool iterating = true;

    while ( iterating )
    {
        if ( boundProtPositions.at(currentIterationIdx) == Cutting )
            boundData.pop_back();

        if ( ( boundProtPositions.at(currentIterationIdx) == Cutting ) &&
             ( iteration.at(currentIterationIdx) == pwl2limodsat::GeqZero ) )
        {
            iteration.at(currentIterationIdx) = pwl2limodsat::LeqZero;
            iterating = false;
        }
        else
        {
            iteration.at(currentIterationIdx) = pwl2limodsat::GeqZero;
            if ( currentIterationIdx > limitIterationIdx )
                currentIterationIdx--;
            else
                return false;
        }
    }

    return true;
}

bool NeuralNetwork::iterate(std::vector<pwl2limodsat::BoundarySymbol>& iteration,
                            size_t& currentIterationIdx,
                            const std::vector<BoundProtPosition>& boundProtPositions,
                            pwl2limodsat::BoundaryCollection& boundData)
{
    return iterate(0, iteration, currentIterationIdx, boundProtPositions, boundData);
}

void NeuralNetwork::net2pwl(pwl2limodsat::BoundaryPrototypeCollection& boundProtData,
                            std::vector<pwl2limodsat::PiecewiseLinearFunctionData>& pwlData,
                            const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                            const pwl2limodsat::BoundaryCollection& currentBoundData,
                            size_t layerNum)
{
    pwl2limodsat::BoundaryPrototypeCollection newBoundProtData;

    if ( layerNum == 0 )
        newBoundProtData = inputValues;
    else
        newBoundProtData = composeBoundProtData(inputValues, layerNum);

    pwl2limodsat::BoundProtIndex newBoundProtDataFirstIdx = boundProtData.size();
    writeBoundProtData(boundProtData, newBoundProtData);

    if ( layerNum + 1 == neuralNetwork.size() )
    {
        for ( size_t outIdx = 0; outIdx < nnOutputIndexes.size(); outIdx++ )
        {
            boundProtData.push_back( boundProtData.at(newBoundProtDataFirstIdx+nnOutputIndexes.at(outIdx)) );
            boundProtData.back().at(0) = boundProtData.back().at(0) - 1;
            writePwlData( pwlData.at(outIdx),
                          boundProtData,
                          newBoundProtData.at( nnOutputIndexes.at(outIdx) ),
                          currentBoundData,
                          std::pair<pwl2limodsat::BoundProtIndex, pwl2limodsat::BoundProtIndex>(newBoundProtDataFirstIdx + nnOutputIndexes.at(outIdx),
                                                                                                boundProtData.size() - 1) );
        }
    }
    else
    {
        std::vector<BoundProtPosition> boundProtPositions;
        std::vector<pwl2limodsat::BoundarySymbol> iteration(newBoundProtData.size(), pwl2limodsat::GeqZero);

        for ( size_t i = newBoundProtDataFirstIdx; i < newBoundProtDataFirstIdx + newBoundProtData.size(); i++ )
            boundProtPositions.push_back( boundProtPosition(boundProtData, i) );

        size_t currentIterationIdx = 0;
        bool iterated = true;
        pwl2limodsat::BoundaryCollection auxCurrentBoundData = currentBoundData;

        while ( iterated )
        {
            while ( ( iterated ) && ( currentIterationIdx < iteration.size() ) )
            {
                if ( boundProtPositions.at(currentIterationIdx) != Cutting )
                    currentIterationIdx++;
                else
                {
                    if ( iteration.at(currentIterationIdx) == pwl2limodsat::GeqZero )
                        auxCurrentBoundData.push_back( pwl2limodsat::Boundary(newBoundProtDataFirstIdx + currentIterationIdx,
                                                                              pwl2limodsat::GeqZero) );
                    else
                        auxCurrentBoundData.push_back( pwl2limodsat::Boundary(newBoundProtDataFirstIdx + currentIterationIdx,
                                                                              pwl2limodsat::LeqZero) );

                    if ( feasibleBounds(boundProtData, auxCurrentBoundData) )
                        currentIterationIdx++;
                    else
                        iterated = iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBoundData);
                }
            }

            if ( iterated )
            {
                pwl2limodsat::BoundaryPrototypeCollection outputValues = composeOutputValues(newBoundProtData,
                                                                                             iteration,
                                                                                             boundProtPositions);

                net2pwl(boundProtData, pwlData, outputValues, auxCurrentBoundData, layerNum+1);

                currentIterationIdx--;
                iterated = iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBoundData);
            }
        }
    }
}

void NeuralNetwork::net2pwl(const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                            const pwl2limodsat::BoundaryCollection& currentBoundData,
                            size_t layerNum)
{
    net2pwl(boundProtData, pwlData, inputValues, currentBoundData, layerNum);
}

std::pair<std::vector<pwl2limodsat::PiecewiseLinearFunctionData>,
          pwl2limodsat::BoundaryPrototypeCollection> NeuralNetwork::partialNet2pwl(unsigned startingPoint,
                                                                                   size_t fixedNodesNum,
                                                                                   const pwl2limodsat::BoundaryPrototypeCollection& inputValues,
                                                                                   const std::vector<BoundProtPosition>& boundProtPositions)
{
    pwl2limodsat::BoundaryPrototypeCollection localBoundProtData;
    std::vector<pwl2limodsat::PiecewiseLinearFunctionData> localPwlData;
    pwl2limodsat::BoundaryPrototypeCollection newBoundProtData = inputValues;

    for ( unsigned outIdx : nnOutputIndexes )
    {
        pwl2limodsat::PiecewiseLinearFunctionData emptyPwlData;
        localPwlData.push_back(emptyPwlData);
    }

    writeBoundProtData(localBoundProtData, newBoundProtData);

    pwl2limodsat::BoundaryCollection auxCurrentBoundData;
    std::vector<pwl2limodsat::BoundarySymbol> iteration;

    size_t fixedNodesIt = 0, fixedNodes = 0;
    while ( fixedNodes < fixedNodesNum )
    {
        if ( boundProtPositions.at(fixedNodesIt) != Cutting )
            iteration.push_back(pwl2limodsat::GeqZero);
        else
        {
            if ( ( ( startingPoint >> fixedNodes ) & 1 ) == 0 )
                iteration.push_back( pwl2limodsat::GeqZero );
            else
                iteration.push_back( pwl2limodsat::LeqZero );

            if ( iteration.back() == pwl2limodsat::GeqZero )
                auxCurrentBoundData.push_back( pwl2limodsat::Boundary(iteration.size() - 1, pwl2limodsat::GeqZero) );
            else
                auxCurrentBoundData.push_back( pwl2limodsat::Boundary(iteration.size() - 1, pwl2limodsat::LeqZero) );

            fixedNodes++;
        }
        fixedNodesIt++;
    }

    size_t minIterationIdx = iteration.size();
    size_t currentIterationIdx = minIterationIdx;

    for ( size_t i = 0; i < boundProtPositions.size() - fixedNodesIt; i++ )
        iteration.push_back(pwl2limodsat::GeqZero);

    bool iterated = true;

    while ( iterated )
    {
        while ( ( iterated ) && ( currentIterationIdx < iteration.size() ) )
        {
            if ( boundProtPositions.at(currentIterationIdx) != Cutting )
                currentIterationIdx++;
            else
            {
                if ( iteration.at(currentIterationIdx) == pwl2limodsat::GeqZero )
                    auxCurrentBoundData.push_back( pwl2limodsat::Boundary(currentIterationIdx, pwl2limodsat::GeqZero) );
                else
                    auxCurrentBoundData.push_back( pwl2limodsat::Boundary(currentIterationIdx, pwl2limodsat::LeqZero) );

                if ( feasibleBounds(boundProtData, auxCurrentBoundData) )
                    currentIterationIdx++;
                else
                    iterated = iterate(minIterationIdx, iteration, currentIterationIdx, boundProtPositions, auxCurrentBoundData);
            }
        }

        if ( iterated )
        {
            pwl2limodsat::BoundaryPrototypeCollection outputValues = composeOutputValues(newBoundProtData,
                                                                                         iteration,
                                                                                         boundProtPositions);

            net2pwl(localBoundProtData, localPwlData, outputValues, auxCurrentBoundData, 1);

            currentIterationIdx--;
            iterated = iterate(minIterationIdx, iteration, currentIterationIdx, boundProtPositions, auxCurrentBoundData);
        }
    }

    return std::pair<std::vector<pwl2limodsat::PiecewiseLinearFunctionData>,
                     pwl2limodsat::BoundaryPrototypeCollection>(localPwlData, localBoundProtData);
}

void NeuralNetwork::pwlInfoMerge(const std::vector<std::pair<std::vector<pwl2limodsat::PiecewiseLinearFunctionData>,
                                                             pwl2limodsat::BoundaryPrototypeCollection>>& threadsInfo)
{
    unsigned boundProtDataInitialSize = boundProtData.size();
    for ( size_t i = 0; i < threadsInfo.size(); i++ )
    {
        unsigned boundProtDataCurrentSize = boundProtData.size();
        boundProtData.insert(boundProtData.end(),
                             threadsInfo.at(i).second.begin() + boundProtDataInitialSize,
                             threadsInfo.at(i).second.end());


        for ( size_t j = 0; j < threadsInfo.at(i).first.size(); j++ )
        {
            for ( size_t k = 0; k < threadsInfo.at(i).first.at(j).size(); k++ )
            {
                pwlData.at(j).push_back(threadsInfo.at(i).first.at(j).at(k));

                if ( i > 0 )
                {
                    for ( size_t l = 0; l < pwlData.at(j).back().bound.size(); l++ )
                        if ( pwlData.at(j).back().bound.at(l).first >= boundProtDataInitialSize )
                        {
                            pwlData.at(j).back().bound.at(l).first += boundProtDataCurrentSize;
                            pwlData.at(j).back().bound.at(l).first -= boundProtDataInitialSize;
                        }
                }
            }
        }
    }
}

void NeuralNetwork::net2pwlMultithreading(const pwl2limodsat::BoundaryPrototypeCollection& inputValues)
{
    pwl2limodsat::BoundaryPrototypeCollection newBoundProtData = inputValues;
    std::vector<BoundProtPosition> boundProtPositions;

    writeBoundProtData(boundProtData, newBoundProtData);

    for ( size_t i = 0; i < newBoundProtData.size(); i++ )
        boundProtPositions.push_back( boundProtPosition(boundProtData, i) );

    unsigned cuttingNodes = 0;
    for ( size_t i = 0; i < boundProtPositions.size(); i++ )
        if ( boundProtPositions.at(i) == Cutting )
            cuttingNodes++;

    size_t fixedNodesMax = floor(log2(std::thread::hardware_concurrency()));
    size_t fixedNodes = ( cuttingNodes > fixedNodesMax ? fixedNodesMax : cuttingNodes );
    unsigned threadsNum = pow(2, fixedNodes);

    std::vector<std::future<std::pair<std::vector<pwl2limodsat::PiecewiseLinearFunctionData>,
                                      pwl2limodsat::BoundaryPrototypeCollection>>> threadsInfoFut;
    for ( unsigned i = 0; i < threadsNum; i++ )
        threadsInfoFut.push_back( async(&NeuralNetwork::partialNet2pwl, this, i, fixedNodes, inputValues, boundProtPositions) );

    std::vector<std::pair<std::vector<pwl2limodsat::PiecewiseLinearFunctionData>,
                          pwl2limodsat::BoundaryPrototypeCollection>> threadsInfo;
    for ( size_t i = 0; i < threadsInfoFut.size(); i++ )
        threadsInfo.push_back(threadsInfoFut.at(i).get());

    pwlInfoMerge(threadsInfo);
}

void NeuralNetwork::net2pwl()
{
    pwl2limodsat::BoundaryPrototypeCollection firstInputValues;

    for ( size_t i = 0; i < neuralNetwork.at(0).size(); i++ )
    {
        pwl2limodsat::BoundaryPrototype auxFirstInputValues;

        for ( size_t j = 0; j < neuralNetwork.at(0).at(0).size(); j++ )
            auxFirstInputValues.push_back(neuralNetwork.at(0).at(i).at(j));

        firstInputValues.push_back(auxFirstInputValues);
    }

    if ( processingMode == Multi )
    {
        net2pwlMultithreading(firstInputValues);
    }
    else if ( processingMode == Single )
    {
        pwl2limodsat::BoundaryCollection emptyBounds;
        net2pwl(firstInputValues, emptyBounds, 0);
    }

    pwlTranslation = true;
}

pwl2limodsat::PiecewiseLinearFunctionData NeuralNetwork::getPwlData(unsigned nnOutputIdx)
{
    size_t outIdx = getNnOutputIndexesIdx(nnOutputIdx);

    if ( !pwlTranslation )
        net2pwl();

    return pwlData.at(outIdx);
}

pwl2limodsat::BoundaryPrototypeCollection NeuralNetwork::getBoundProtData()
{
    if ( !pwlTranslation )
        net2pwl();

    return boundProtData;
}

void NeuralNetwork::printPwlFile(unsigned nnOutputIdx)
{
    size_t outIdx = getNnOutputIndexesIdx(nnOutputIdx);

    std::ofstream pwlFile(pwlFileName.at(outIdx));

    if ( !pwlTranslation )
        net2pwl();

    pwlFile << "pwl" << std::endl << std::endl;

    for ( size_t i = 0; i < boundProtData.size(); i++ )
    {
        pwlFile << "b ";

        for ( size_t j = 0; j < boundProtData.at(0).size(); j++ )
        {
            pwlFile << boundProtData.at(i).at(j);

            if ( j + 1 != boundProtData.at(0).size() )
                pwlFile << " ";
        }

        pwlFile << std::endl;
    }

    for ( size_t i = 0; i < pwlData.at(outIdx).size(); i++ )
    {
        pwlFile << std::endl << "p ";

        for ( size_t j = 0; j < pwlData.at(outIdx).at(i).lpData.size(); j++ )
        {
            pwlFile << pwlData.at(outIdx).at(i).lpData.at(j).first << " " << pwlData.at(outIdx).at(i).lpData.at(j).second;

            if ( j+1 != pwlData.at(outIdx).at(i).lpData.size() )
                pwlFile << " ";
            else
                pwlFile << std::endl;
        }

        for ( size_t j = 0; j < pwlData.at(outIdx).at(i).bound.size(); j++ )
        {
            if ( pwlData.at(outIdx).at(i).bound.at(j).second == pwl2limodsat::GeqZero )
                pwlFile << "g ";
            else
                pwlFile << "l ";

            pwlFile << pwlData.at(outIdx).at(i).bound.at(j).first + 1;

            if ( j+1 != pwlData.at(outIdx).at(i).bound.size() )
                pwlFile << std::endl;
        }

        if ( i+1 != pwlData.at(outIdx).size() )
            pwlFile << std::endl;
    }
}
}
