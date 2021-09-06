#include <fstream>
#include <thread>
#include "soplex.h"
#include "NeuralNetwork.h"

#define PRECISION 1000000

using namespace std;
using namespace soplex;

NeuralNetwork::NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName, bool mth) :
    net(inputNet), multithreading(mth)
{
    if ( onnxFileName.substr(onnxFileName.size()-5,5) == ".onnx" )
        pwlFileName = onnxFileName.substr(0,onnxFileName.size()-5);
    else
        pwlFileName = onnxFileName;

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

LinearPieceCoefficient NeuralNetwork::dec2frac(const NodeCoeff& decValue)
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

BoundProtPosition NeuralNetwork::boundProtPosition(BoundProtIndex bIdx)
{
    SoPlex sop;

    BoundaryCoefficient K = -boundProts.at(bIdx).at(0);

    DSVector dummycol(0);
    for ( auto i = 1; i < boundProts.at(bIdx).size(); i++ )
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

bool NeuralNetwork::feasibleBoundsWithExtra(const vector<Boundary>& bounds, const vector<Boundary>& extraBounds)
{
    SoPlex sop;

    DSVector dummycol(0);
    for ( auto i = 1; i <= net.at(0).at(0).size()-1; i++ )
        sop.addColReal(LPCol(0, dummycol, 1, 0));

    DSVector row(net.at(0).size());
    for ( auto i = 0; i < bounds.size(); i++ )
    {
        for ( auto j = 1; j <= net.at(0).at(0).size()-1; j++ )
            row.add(j-1, boundProts.at(bounds.at(i).first).at(j));

        if ( bounds.at(i).second == GeqZero )
            sop.addRowReal(LPRow(-boundProts.at(bounds.at(i).first).at(0), row, infinity));
        else if ( bounds.at(i).second == LeqZero )
            sop.addRowReal(LPRow(-infinity, row, -boundProts.at(bounds.at(i).first).at(0)));

        row.clear();
    }
    for ( auto i = 0; i < extraBounds.size(); i++ )
    {
        for ( auto j = 1; j <= net.at(0).at(0).size()-1; j++ )
            row.add(j-1, boundProts.at(extraBounds.at(i).first).at(j));

        if ( extraBounds.at(i).second == GeqZero )
            sop.addRowReal(LPRow(-boundProts.at(extraBounds.at(i).first).at(0), row, infinity));
        else if ( extraBounds.at(i).second == LeqZero )
            sop.addRowReal(LPRow(-infinity, row, -boundProts.at(extraBounds.at(i).first).at(0)));

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

bool NeuralNetwork::feasibleBounds(const vector<Boundary>& bounds)
{
    vector<Boundary> dummyExtraBounds;
    return feasibleBoundsWithExtra(bounds, dummyExtraBounds);
}

vector<BoundaryPrototype> NeuralNetwork::composeBoundProts(const vector<BoundaryPrototype>& inputValues, unsigned layerNum)
{
    vector<BoundaryPrototype> newBoundProts;

    for ( auto i = 0; i < net.at(layerNum).size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < inputValues.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;

            if ( j == 0 )
                auxCoeff = net.at(layerNum).at(i).at(0);
            else
                auxCoeff = 0;

            for ( auto k = 0; k < inputValues.size(); k++ )
                auxCoeff = auxCoeff + inputValues.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

            auxBoundProt.push_back(auxCoeff);
        }

        newBoundProts.push_back(auxBoundProt);
    }

    return newBoundProts;
}

void NeuralNetwork::writeBoundProts(const vector<BoundaryPrototype>& newBoundProts)
{
    for ( auto i = 0; i < newBoundProts.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < newBoundProts.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = newBoundProts.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }
}

void NeuralNetwork::writeRegionalLinearPieces(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, BoundProtIndex newBoundProtFirstIdx)
{
    RegionalLinearPiece rlp0;
    rlp0.bounds = currentBounds;
    rlp0.bounds.push_back(Boundary(newBoundProtFirstIdx, LeqZero));
    if ( feasibleBounds(rlp0.bounds) )
    {
        for ( auto i = 0; i < inputValues.at(0).size(); i++ )
            rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
        rlPieces.push_back(rlp0);
    }

    RegionalLinearPiece rlp0_1;
    rlp0_1.bounds = currentBounds;
    rlp0_1.bounds.push_back(Boundary(newBoundProtFirstIdx, GeqZero));
    rlp0_1.bounds.push_back(Boundary(newBoundProtFirstIdx+1, LeqZero));
    if ( feasibleBounds(rlp0_1.bounds) )
    { //for ( auto i = 0; i < rlp0_1.bounds.size(); i++ ) cout << rlp0_1.bounds.at(i).first << " " << rlp0_1.bounds.at(i).second << " | "; cout << endl;
        for ( auto i = 0; i < inputValues.at(0).size(); i++ )
            rlp0_1.coefs.push_back(dec2frac(inputValues.at(0).at(i)));
        rlPieces.push_back(rlp0_1);
    }

    RegionalLinearPiece rlp1;
    rlp1.bounds = currentBounds;
    rlp1.bounds.push_back(Boundary(newBoundProtFirstIdx+1, GeqZero));
    if ( feasibleBounds(rlp1.bounds) )
    {
        rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
        for ( auto i = 1; i < inputValues.at(0).size(); i++ )
            rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
        rlPieces.push_back(rlp1);
    }
}

void NeuralNetwork::iterate(vector<unsigned>& iteration, int& currentIterationIdx, const vector<BoundProtPosition>& boundProtPositions, vector<Boundary>& bounds)
{
    bool iterating = true;

    while ( iterating && ( currentIterationIdx >= 0 ) )
    {
        if ( boundProtPositions.at(currentIterationIdx) == Cutting )
            bounds.pop_back();

        if ( ( boundProtPositions.at(currentIterationIdx) == Cutting ) && ( iteration.at(currentIterationIdx) == 0 ) )
        {
            iteration.at(currentIterationIdx)++;
            iterating = false;
        }
        else
        {
            iteration.at(currentIterationIdx) = 0;
            currentIterationIdx--;
        }
    }
}

void NeuralNetwork::net2pwl(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<BoundaryPrototype> newBoundProts;

    if ( layerNum == 0 )
        newBoundProts = inputValues;
    else
        newBoundProts = composeBoundProts(inputValues, layerNum);

    BoundProtIndex newBoundProtFirstIdx = boundProts.size();
    writeBoundProts(newBoundProts);

    if ( layerNum + 1 == net.size() )
    {
        boundProts.push_back(boundProts.at(newBoundProtFirstIdx));
        boundProts.at(newBoundProtFirstIdx+1).at(0) = boundProts.at(newBoundProtFirstIdx+1).at(0) - 1;

        writeRegionalLinearPieces(newBoundProts, currentBounds, newBoundProtFirstIdx);
    }
    else
    {
        vector<BoundProtPosition> boundProtPositions;
        vector<unsigned> iteration(newBoundProts.size(), 0);

        for ( auto i = newBoundProtFirstIdx; i < newBoundProtFirstIdx + newBoundProts.size(); i++ )
            boundProtPositions.push_back( boundProtPosition(i) );

        int currentIterationIdx = 0;
        vector<Boundary> auxCurrentBounds = currentBounds;

        while ( currentIterationIdx >= 0 )
        {
            vector<BoundaryPrototype> outputValues = newBoundProts;

            while ( ( currentIterationIdx < iteration.size() ) && ( currentIterationIdx >= 0 ) )
            {
                if ( boundProtPositions.at(currentIterationIdx) != Cutting )
                    currentIterationIdx++;
                else
                {
                    if ( iteration.at(currentIterationIdx) == 1 )
                        auxCurrentBounds.push_back( Boundary(newBoundProtFirstIdx+currentIterationIdx,GeqZero) );
                    else
                        auxCurrentBounds.push_back( Boundary(newBoundProtFirstIdx+currentIterationIdx,LeqZero) );

                    if ( feasibleBounds( auxCurrentBounds ) )
                        currentIterationIdx++;
                    else
                        iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
                }
            }

            if ( currentIterationIdx >= 0 )
            {
                for ( auto j = 0; j < iteration.size(); j++ )
                {
                    if ( boundProtPositions.at(j) == Cutting )
                    {
                        if ( iteration.at(j) == 0 )
                            for ( auto k = 0; k < outputValues.at(0).size(); k++ )
                                outputValues.at(j).at(k) = 0;
                    }
                    else if ( boundProtPositions.at(j) == Under )
                    {
                        for ( auto k = 0; k < outputValues.at(0).size(); k++ )
                            outputValues.at(j).at(k) = 0;
                    }
                }

                net2pwl(outputValues, auxCurrentBounds, layerNum+1);

                currentIterationIdx--;
                iterate(iteration, currentIterationIdx, boundProtPositions, auxCurrentBounds);
            }
        }
    }
}

void NeuralNetwork::iterate_mthreading(const vector<BoundaryPrototype>& newBoundProts, const vector<Boundary>& currentBounds, vector<Boundary> tBounds, BoundProtIndex fbpIdx, const vector<BoundProtPosition>& boundProtPositions, unsigned posIdx, unsigned layerNum)
{
    if ( posIdx < boundProtPositions.size() )
    {
        if ( boundProtPositions.at(posIdx) == Cutting )
        {
            tBounds.push_back( Boundary(fbpIdx+posIdx, GeqZero) );

            thread t1;
            boundProtsMutex.lock();
            if ( feasibleBoundsWithExtra(currentBounds, tBounds) )
                t1 = thread(&NeuralNetwork::iterate_mthreading, this, newBoundProts, currentBounds, tBounds, fbpIdx, boundProtPositions, posIdx+1, layerNum);
            boundProtsMutex.unlock();

            tBounds.pop_back();
            tBounds.push_back( Boundary(fbpIdx+posIdx, LeqZero) );

            thread t2;
            boundProtsMutex.lock();
            if ( feasibleBoundsWithExtra(currentBounds, tBounds) )
                t2 = thread(&NeuralNetwork::iterate_mthreading, this, newBoundProts, currentBounds, tBounds, fbpIdx, boundProtPositions, posIdx+1, layerNum);
            boundProtsMutex.unlock();

            if ( t1.joinable() )
                t1.join();
            if ( t2.joinable() )
                t2.join();
        }
        else
            iterate_mthreading(newBoundProts, currentBounds, tBounds, fbpIdx, boundProtPositions, posIdx+1, layerNum);
    }
    else
    {
        vector<BoundaryPrototype> outputValues = newBoundProts;
            unsigned fBoundsPos = 0;

            for ( auto j = 0; j < boundProtPositions.size(); j++ )
            {
                if ( boundProtPositions.at(j) == Cutting )
                {
                    if ( tBounds.at(fBoundsPos).second == LeqZero )
                        for ( auto k = 0; k < outputValues.at(0).size(); k++ )
                            outputValues.at(j).at(k) = 0;

                    fBoundsPos++;
                }
                else if ( boundProtPositions.at(j) == Under )
                {
                    for ( auto k = 0; k < outputValues.at(0).size(); k++ )
                        outputValues.at(j).at(k) = 0;
                }
            }

            vector<Boundary> auxCurrentBounds(currentBounds);
            auxCurrentBounds.insert(auxCurrentBounds.end(), tBounds.begin(), tBounds.end());

            net2pwl_mthreading(outputValues, auxCurrentBounds, layerNum+1);
    }
}

void NeuralNetwork::net2pwl_mthreading(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<BoundaryPrototype> newBoundProts;

    if ( layerNum == 0 )
        newBoundProts = inputValues;
    else
        newBoundProts = composeBoundProts(inputValues, layerNum);

    boundProtsMutex.lock();
    BoundProtIndex newBoundProtFirstIdx = boundProts.size();
    writeBoundProts(newBoundProts);
    if ( layerNum + 1 == net.size() )
    {
        boundProts.push_back(boundProts.at(newBoundProtFirstIdx));
        boundProts.at(newBoundProtFirstIdx+1).at(0) = boundProts.at(newBoundProtFirstIdx+1).at(0) - 1;
    }
    boundProtsMutex.unlock();

    if ( layerNum + 1 == net.size() )
    {
        boundProtsMutex.lock();

        rlPiecesMutex.lock();
        writeRegionalLinearPieces(newBoundProts, currentBounds, newBoundProtFirstIdx);
        rlPiecesMutex.unlock();

        boundProtsMutex.unlock();
    }
    else
    {
        vector<BoundProtPosition> boundProtPositions;

        boundProtsMutex.lock();
        for ( auto i = newBoundProtFirstIdx; i < newBoundProtFirstIdx + newBoundProts.size(); i++ )
            boundProtPositions.push_back( boundProtPosition(i) );
        boundProtsMutex.unlock();

        vector<Boundary> dummyBounds;

        iterate_mthreading(newBoundProts, currentBounds, dummyBounds, newBoundProtFirstIdx, boundProtPositions, 0, layerNum);
    }
}

void NeuralNetwork::net2pwl()
{
    vector<BoundaryPrototype> firstInputValues;

    for ( auto i = 0; i < net.at(0).size(); i++ )
    {
        BoundaryPrototype auxFirstInputValues;

        for ( auto j = 0; j < net.at(0).at(0).size(); j++ )
            auxFirstInputValues.push_back(net.at(0).at(i).at(j));

        firstInputValues.push_back(auxFirstInputValues);
    }

    vector<Boundary> emptyBounds;

    if ( multithreading )
        net2pwl_mthreading(firstInputValues, emptyBounds, 0);
    else
        net2pwl(firstInputValues, emptyBounds, 0);

    pwlTranslation = true;
}

vector<RegionalLinearPiece> NeuralNetwork::getRegLinPieces()
{
    if ( !pwlTranslation )
        net2pwl();

    return rlPieces;
}

vector<BoundaryPrototype> NeuralNetwork::getBoundPrototypes()
{
    if ( !pwlTranslation )
        net2pwl();

    return boundProts;
}

void NeuralNetwork::printPwlFile()
{ //cout << "CHEGOU AQUIII ";
    ofstream pwlFile(pwlFileName);

    if ( !pwlTranslation )
        net2pwl();

    pwlFile << "pwl" << endl << endl;

    for ( auto i = 0; i < boundProts.size(); i++ )
    {
        pwlFile << "b ";

        for ( auto j = 0; j < boundProts.at(0).size(); j++ )
        {
            pwlFile << boundProts.at(i).at(j);

            if ( j + 1 != boundProts.at(0).size() )
                pwlFile << " ";
        }

        pwlFile << endl;
    }

    for ( auto i = 0; i < rlPieces.size(); i++ )
    {
        pwlFile << endl << "p ";

        for ( auto j = 0; j < rlPieces.at(i).coefs.size(); j++ )
        {
            pwlFile << rlPieces.at(i).coefs.at(j).first << " " << rlPieces.at(i).coefs.at(j).second;

            if ( j+1 != rlPieces.at(i).coefs.size() )
                pwlFile << " ";
            else
                pwlFile << endl;
        }

        for ( auto j = 0; j < rlPieces.at(i).bounds.size(); j++ )
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
    } //cout << " E AQUI!!" << endl;
}
