#include <fstream>
#include <cmath>
#include <future>
#include "soplex.h"
#include "NeuralNetwork.h"

#define PRECISION 1000000

using namespace std;
using namespace soplex;

NeuralNetwork::NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName, bool multithreading) :
    net(inputNet)
{
    if ( onnxFileName.substr(onnxFileName.size()-5,5) == ".onnx" )
        pwlFileName = onnxFileName.substr(0,onnxFileName.size()-5);
    else
        pwlFileName = onnxFileName;

    processingMode = ( multithreading ? Multi : Single );

    pwlFileName.append(".pwl");
}

NeuralNetwork::NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName) :
    NeuralNetwork(inputNet, onnxFileName, true) {}

lpcNonNegative NeuralNetwork::gcd(lpcNonNegative a, lpcNonNegative b)
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

LinearPieceCoefficient NeuralNetwork::dec2frac(NodeCoeff decValue)
{
    lpcNonNegative whole;
    bool negFactor = false;

    if ( decValue >= 0 )
        whole = floor(decValue);
    else
    {
        whole = abs(ceil(decValue));
        negFactor = true;
    }

    lpcNonNegative decimals = ( (negFactor ? -1 : 1) * decValue - whole ) * PRECISION;
    decimals = decimals + whole * PRECISION;
    lpcNonNegative factor = gcd(decimals, PRECISION);

    lpcNonNegative denominator = PRECISION / factor;
    lpcInteger numerator = ( decimals / factor ) * (negFactor ? -1 : 1);

    return LinearPieceCoefficient(numerator, denominator);
}

BoundProtPosition NeuralNetwork::boundProtPosition(const vector<BoundaryPrototype>& boundProts, BoundProtIndex bIdx)
{
    SoPlex sop;

    BoundaryCoefficient K = -boundProts.at(bIdx).at(0);

    DSVector dummycol(0);
    for ( size_t i = 1; i < boundProts.at(bIdx).size(); i++ )
        sop.addColReal(LPCol(boundProts.at(bIdx).at(i), dummycol, 1, 0));

    sop.setIntParam(SoPlex::VERBOSITY, SoPlex::VERBOSITY_ERROR);
    sop.setIntParam(SoPlex::OBJSENSE, SoPlex::OBJSENSE_MAXIMIZE);
    sop.optimize();
    float Max = sop.objValueReal();

    sop.setIntParam(SoPlex::OBJSENSE, SoPlex::OBJSENSE_MINIMIZE);
    sop.optimize();
    float Min = sop.objValueReal();

    if ( (Min >= K) && (Max >= K) )
        return Over;
    else if ( (Max <= K) && (Min <= K) )
        return Under;
    else
        return Cutting;
}

bool NeuralNetwork::feasibleBounds(const vector<BoundaryPrototype>& boundProts, const vector<Boundary>& bounds)
{
    SoPlex sop;

    DSVector dummycol(0);
    for ( size_t i = 1; i <= net.at(0).at(0).size()-1; i++ )
        sop.addColReal(LPCol(0, dummycol, 1, 0));

    DSVector row(net.at(0).size());
    for ( size_t i = 0; i < bounds.size(); i++ )
    {
        for ( size_t j = 1; j <= net.at(0).at(0).size()-1; j++ )
            row.add(j-1, boundProts.at(bounds.at(i).first).at(j));

        if ( bounds.at(i).second == GeqZero )
            sop.addRowReal(LPRow(-boundProts.at(bounds.at(i).first).at(0), row, infinity));
        else if ( bounds.at(i).second == LeqZero )
            sop.addRowReal(LPRow(-infinity, row, -boundProts.at(bounds.at(i).first).at(0)));

        row.clear();
    }

    sop.setIntParam(SoPlex::VERBOSITY, SoPlex::VERBOSITY_ERROR);
    sop.setIntParam(SoPlex::OBJSENSE, SoPlex::OBJSENSE_MAXIMIZE);
    sop.optimize();
    float Max = sop.objValueReal();

    if ( Max < 0 )
        return false;
    else
        return true;
}

vector<BoundaryPrototype> NeuralNetwork::composeBoundProts(const vector<BoundaryPrototype>& inputValues, unsigned layerNum)
{
    vector<BoundaryPrototype> newBoundProts;

    for ( size_t i = 0; i < net.at(layerNum).size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( size_t j = 0; j < inputValues.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;

            if ( j == 0 )
                auxCoeff = net.at(layerNum).at(i).at(0);
            else
                auxCoeff = 0;

            for ( size_t k = 0; k < inputValues.size(); k++ )
                auxCoeff = auxCoeff + inputValues.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

            auxBoundProt.push_back(auxCoeff);
        }

        newBoundProts.push_back(auxBoundProt);
    }

    return newBoundProts;
}

void NeuralNetwork::writeBoundProts(vector<BoundaryPrototype>& boundProts, const vector<BoundaryPrototype>& newBoundProts)
{
    for ( size_t i = 0; i < newBoundProts.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( size_t j = 0; j < newBoundProts.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = newBoundProts.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }
}

vector<BoundaryPrototype> NeuralNetwork::composeOutputValues(const vector<BoundaryPrototype>& boundProts,
                                                             const vector<BoundarySymbol>& iteration,
                                                             const vector<BoundProtPosition>& boundProtPositions)
{
    vector<BoundaryPrototype> outputValues = boundProts;

    for ( size_t j = 0; j < iteration.size(); j++ )
    {
        if ( boundProtPositions.at(j) == Cutting )
        {
            if ( iteration.at(j) == LeqZero )
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

void NeuralNetwork::writeRegionalLinearPieces(vector<RegionalLinearPiece>& rlPieces,
                                              const vector<BoundaryPrototype>& boundProts,
                                              const vector<BoundaryPrototype>& inputValues,
                                              const vector<Boundary>& currentBounds,
                                              BoundProtIndex newBoundProtFirstIdx)
{
    RegionalLinearPiece rlp0;
    rlp0.bounds = currentBounds;
    rlp0.bounds.push_back(Boundary(newBoundProtFirstIdx, LeqZero));
    if ( feasibleBounds(boundProts, rlp0.bounds) )
    {
        for ( size_t i = 0; i < inputValues.at(0).size(); i++ )
            rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
        rlPieces.push_back(rlp0);
    }

    RegionalLinearPiece rlp0_1;
    rlp0_1.bounds = currentBounds;
    rlp0_1.bounds.push_back(Boundary(newBoundProtFirstIdx, GeqZero));
    rlp0_1.bounds.push_back(Boundary(newBoundProtFirstIdx+1, LeqZero));
    if ( feasibleBounds(boundProts, rlp0_1.bounds) )
    {
        for ( size_t i = 0; i < inputValues.at(0).size(); i++ )
            rlp0_1.coefs.push_back(dec2frac(inputValues.at(0).at(i)));
        rlPieces.push_back(rlp0_1);
    }

    RegionalLinearPiece rlp1;
    rlp1.bounds = currentBounds;
    rlp1.bounds.push_back(Boundary(newBoundProtFirstIdx+1, GeqZero));

    if ( feasibleBounds(boundProts, rlp1.bounds) )
    {
        rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
        for ( size_t i = 1; i < inputValues.at(0).size(); i++ )
            rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
        rlPieces.push_back(rlp1);
    }
}

bool NeuralNetwork::iterate(size_t limitIterationIdx,
                            vector<BoundarySymbol>& iteration,
                            size_t& currentIterationIdx,
                            const vector<BoundProtPosition>& boundProtPositions,
                            vector<Boundary>& bounds)
{
    bool iterating = true;

    while ( iterating )
    {
        if ( boundProtPositions.at(currentIterationIdx) == Cutting )
            bounds.pop_back();

        if ( ( boundProtPositions.at(currentIterationIdx) == Cutting ) && ( iteration.at(currentIterationIdx) == GeqZero ) )
        {
            iteration.at(currentIterationIdx) = LeqZero;
            iterating = false;
        }
        else
        {
            iteration.at(currentIterationIdx) = GeqZero;
            if ( currentIterationIdx > limitIterationIdx )
                currentIterationIdx--;
            else
                return false;
        }
    }

    return true;
}

bool NeuralNetwork::iterate(vector<BoundarySymbol>& iteration,
                            size_t& currentIterationIdx,
                            const vector<BoundProtPosition>& boundProtPositions,
                            vector<Boundary>& bounds)
{
    return iterate(0, iteration, currentIterationIdx, boundProtPositions, bounds);
}

void NeuralNetwork::net2pwl(vector<BoundaryPrototype>& boundProts,
                            vector<RegionalLinearPiece>& rlPieces,
                            const vector<BoundaryPrototype>& inputValues,
                            const vector<Boundary>& currentBounds,
                            size_t layerNum)
{
    vector<BoundaryPrototype> newBoundProts;

    if ( layerNum == 0 )
        newBoundProts = inputValues;
    else
        newBoundProts = composeBoundProts(inputValues, layerNum);

    BoundProtIndex newBoundProtsFirstIdx = boundProts.size();
    writeBoundProts(boundProts, newBoundProts);

    if ( layerNum + 1 == net.size() )
    {
        boundProts.push_back(boundProts.at(newBoundProtsFirstIdx));
        boundProts.at(newBoundProtsFirstIdx + 1).at(0) = boundProts.at(newBoundProtsFirstIdx + 1).at(0) - 1;

        writeRegionalLinearPieces(rlPieces, boundProts, newBoundProts, currentBounds, newBoundProtsFirstIdx);
    }
    else
    {
        vector<BoundProtPosition> boundProtPositions;
        vector<BoundarySymbol> iteration(newBoundProts.size(), GeqZero);

        for ( size_t i = newBoundProtsFirstIdx; i < newBoundProtsFirstIdx + newBoundProts.size(); i++ )
            boundProtPositions.push_back( boundProtPosition(boundProts, i) );

        size_t currentIterationIdx = 0;
        bool iterated = true;
        vector<Boundary> auxCurrentBounds = currentBounds;

        while ( iterated )
        {
            while ( ( iterated ) && ( currentIterationIdx < iteration.size() ) )
            {
                if ( boundProtPositions.at(currentIterationIdx) != Cutting )
                    currentIterationIdx++;
                else
                {
                    if ( iteration.at(currentIterationIdx) == GeqZero )
                        auxCurrentBounds.push_back( Boundary(newBoundProtsFirstIdx + currentIterationIdx, GeqZero) );
                    else
                        auxCurrentBounds.push_back( Boundary(newBoundProtsFirstIdx + currentIterationIdx, LeqZero) );

                    if ( feasibleBounds(boundProts, auxCurrentBounds) )
                        currentIterationIdx++;
                    else
                        iterated = iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
                }
            }

            if ( iterated )
            {
                vector<BoundaryPrototype> outputValues = composeOutputValues(newBoundProts, iteration, boundProtPositions);

                net2pwl(boundProts, rlPieces, outputValues, auxCurrentBounds, layerNum+1);

                currentIterationIdx--;
                iterated = iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
            }
        }
    }
}

void NeuralNetwork::net2pwl(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, size_t layerNum)
{
    net2pwl(boundProts, rlPieces, inputValues, currentBounds, layerNum);
}

PwlPartialInfo NeuralNetwork::partialNet2pwl(unsigned startingPoint,
                                             size_t fixedNodesNum,
                                             const vector<BoundaryPrototype>& inputValues,
                                             const vector<BoundProtPosition>& boundProtPositions)
{
    PwlBoundaryPrototypes localBoundProts;
    PwlRegionalLinearPieces localRegionalLinearPieces;
    vector<BoundaryPrototype> newBoundProts = inputValues;

    writeBoundProts(localBoundProts, newBoundProts);

    vector<Boundary> auxCurrentBounds;
    vector<BoundarySymbol> iteration;

    size_t fixedNodesIt = 0, fixedNodes = 0;
    while ( fixedNodes < fixedNodesNum )
    {
        if ( boundProtPositions.at(fixedNodesIt) != Cutting )
            iteration.push_back(GeqZero);
        else
        {
            if ( ( ( startingPoint >> fixedNodes ) & 1 ) == 0 )
                iteration.push_back( GeqZero );
            else
                iteration.push_back( LeqZero );

            if ( iteration.back() == GeqZero )
                auxCurrentBounds.push_back( Boundary(iteration.size() - 1, GeqZero) );
            else
                auxCurrentBounds.push_back( Boundary(iteration.size() - 1, LeqZero) );

            fixedNodes++;
        }
        fixedNodesIt++;
    }

    size_t minIterationIdx = iteration.size();
    size_t currentIterationIdx = minIterationIdx;

    for ( size_t i = 0; i < boundProtPositions.size() - fixedNodesIt; i++ )
        iteration.push_back(GeqZero);

    bool iterated = true;

    while ( iterated )
    {
        while ( ( iterated ) && ( currentIterationIdx < iteration.size() ) )
        {
            if ( boundProtPositions.at(currentIterationIdx) != Cutting )
                currentIterationIdx++;
            else
            {
                if ( iteration.at(currentIterationIdx) == GeqZero )
                    auxCurrentBounds.push_back( Boundary(currentIterationIdx, GeqZero) );
                else
                    auxCurrentBounds.push_back( Boundary(currentIterationIdx, LeqZero) );

                if ( feasibleBounds(boundProts, auxCurrentBounds) )
                    currentIterationIdx++;
                else
                    iterated = iterate(minIterationIdx, iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
            }
        }

        if ( iterated )
        {
            vector<BoundaryPrototype> outputValues = composeOutputValues(newBoundProts, iteration, boundProtPositions);

            net2pwl(localBoundProts, localRegionalLinearPieces, outputValues, auxCurrentBounds, 1);

            currentIterationIdx--;
            iterated = iterate(minIterationIdx, iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
        }
    }

    return PwlPartialInfo(localRegionalLinearPieces, localBoundProts);
}

void NeuralNetwork::pwlInfoMerge(const vector<PwlPartialInfo>& threadsInfo)
{
    unsigned boundProtsInitialSize = boundProts.size();
    for ( size_t i = 0; i < threadsInfo.size(); i++ )
    {
        unsigned boundProtsCurrentSize = boundProts.size();
        boundProts.insert(boundProts.end(), threadsInfo.at(i).second.begin() + boundProtsInitialSize, threadsInfo.at(i).second.end());

        for ( size_t j = 0; j < threadsInfo.at(i).first.size(); j++ )
        {
            rlPieces.push_back(threadsInfo.at(i).first.at(j));

            if ( i > 0 )
            {
                for ( size_t k = 0; k < rlPieces.back().bounds.size(); k++ )
                    if ( rlPieces.back().bounds.at(k).first >= boundProtsInitialSize )
                    {
                        rlPieces.back().bounds.at(k).first += boundProtsCurrentSize;
                        rlPieces.back().bounds.at(k).first -= boundProtsInitialSize;
                    }
            }
        }
    }
}

void NeuralNetwork::net2pwlMultithreading(const vector<BoundaryPrototype>& inputValues)
{
    vector<BoundaryPrototype> newBoundProts = inputValues;
    vector<BoundProtPosition> boundProtPositions;

    writeBoundProts(boundProts, newBoundProts);

    for ( size_t i = 0; i < newBoundProts.size(); i++ )
        boundProtPositions.push_back( boundProtPosition(boundProts, i) );

    unsigned cuttingNodes = 0;
    for ( size_t i = 0; i < boundProtPositions.size(); i++ )
        if ( boundProtPositions.at(i) == Cutting )
            cuttingNodes++;

    size_t fixedNodesMax = floor(log2(thread::hardware_concurrency()));
    size_t fixedNodes = ( cuttingNodes > fixedNodesMax ? fixedNodesMax : cuttingNodes );
    unsigned threadsNum = pow(2, fixedNodes);

    if ( threadsNum > 1 )
        cout << "(" << threadsNum << " simultaneous threads)" << endl;
    else
        cout << "(Only one thread was necessary or possible)" << endl;

    vector<future<PwlPartialInfo>> threadsInfoFut;
    for ( unsigned i = 0; i < threadsNum; i++ )
        threadsInfoFut.push_back( async(&NeuralNetwork::partialNet2pwl, this, i, fixedNodes, inputValues, boundProtPositions) );

    vector<PwlPartialInfo> threadsInfo;
    for ( size_t i = 0; i < threadsInfoFut.size(); i++ )
        threadsInfo.push_back(threadsInfoFut.at(i).get());

    pwlInfoMerge(threadsInfo);
}

void NeuralNetwork::net2pwl()
{
    vector<BoundaryPrototype> firstInputValues;

    for ( size_t i = 0; i < net.at(0).size(); i++ )
    {
        BoundaryPrototype auxFirstInputValues;

        for ( size_t j = 0; j < net.at(0).at(0).size(); j++ )
            auxFirstInputValues.push_back(net.at(0).at(i).at(j));

        firstInputValues.push_back(auxFirstInputValues);
    }

    if ( processingMode == Multi )
    {
        cout << "Entering multithreading mode..." << endl;
        net2pwlMultithreading(firstInputValues);
    }
    else if ( processingMode == Single )
    {
        cout << "Working with a single thread..." << endl;
        vector<Boundary> emptyBounds;
        net2pwl(firstInputValues, emptyBounds, 0);
    }

    pwlTranslation = true;
}

PwlRegionalLinearPieces NeuralNetwork::getRegLinPieces()
{
    if ( !pwlTranslation )
        net2pwl();

    return rlPieces;
}

PwlBoundaryPrototypes NeuralNetwork::getBoundPrototypes()
{
    if ( !pwlTranslation )
        net2pwl();

    return boundProts;
}

void NeuralNetwork::printPwlFile()
{
    ofstream pwlFile(pwlFileName);

    if ( !pwlTranslation )
        net2pwl();

    pwlFile << "pwl" << endl << endl;

    for ( size_t i = 0; i < boundProts.size(); i++ )
    {
        pwlFile << "b ";

        for ( size_t j = 0; j < boundProts.at(0).size(); j++ )
        {
            pwlFile << boundProts.at(i).at(j);

            if ( j + 1 != boundProts.at(0).size() )
                pwlFile << " ";
        }

        pwlFile << endl;
    }

    for ( size_t i = 0; i < rlPieces.size(); i++ )
    {
        pwlFile << endl << "p ";

        for ( size_t j = 0; j < rlPieces.at(i).coefs.size(); j++ )
        {
            pwlFile << rlPieces.at(i).coefs.at(j).first << " " << rlPieces.at(i).coefs.at(j).second;

            if ( j+1 != rlPieces.at(i).coefs.size() )
                pwlFile << " ";
            else
                pwlFile << endl;
        }

        for ( size_t j = 0; j < rlPieces.at(i).bounds.size(); j++ )
        {
            if ( rlPieces.at(i).bounds.at(j).second == GeqZero )
                pwlFile << "g ";
            else
                pwlFile << "l ";

            pwlFile << rlPieces.at(i).bounds.at(j).first + 1;

            if ( j+1 != rlPieces.at(i).bounds.size() )
                pwlFile << endl;
        }

        if ( i+1 != rlPieces.size() )
            pwlFile << endl;
    }
}
