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

void NeuralNetwork::iterate_mthreading(vector<vector<Boundary>>& fBounds, const vector<Boundary>& cBounds, vector<Boundary> tBounds, BoundProtIndex fbpIdx, const vector<BoundProtPosition>& pos, unsigned posIdx)
{
    if ( posIdx < pos.size() )
    {
        if ( pos.at(posIdx) == Cutting )
        {
            tBounds.push_back( Boundary(fbpIdx+posIdx, GeqZero) );

            thread t1;
            if ( feasibleBoundsWithExtra(cBounds, tBounds) )
                t1 = thread(&NeuralNetwork::iterate_mthreading, this, ref(fBounds), cBounds, tBounds, fbpIdx, pos, posIdx+1);

            tBounds.pop_back();
            tBounds.push_back( Boundary(fbpIdx+posIdx, LeqZero) );

            thread t2;
            if ( feasibleBoundsWithExtra(cBounds, tBounds) )
                t2 = thread(&NeuralNetwork::iterate_mthreading, this, ref(fBounds), cBounds, tBounds, fbpIdx, pos, posIdx+1);

            if ( t1.joinable() )
                t1.join();
            if ( t2.joinable() )
                t2.join();
        }
        else
            iterate_mthreading(fBounds, cBounds, tBounds, fbpIdx, pos, posIdx+1);
    }
    else
    {
        multithreadingMutex.lock();
        fBounds.push_back(tBounds);
        multithreadingMutex.unlock();
    }
}

void NeuralNetwork::net2pwl_mthreading(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<Node> nextInfo;

    if ( layerNum == 0 )
        nextInfo = previousInfo;
    else
    {
        for ( auto i = 0; i < net.at(layerNum).size(); i++ )
        {
            Node auxNode;

            for ( auto j = 0; j < previousInfo.at(0).size(); j++ )
            {
                NodeCoeff auxCoeff;

                if ( j == 0 )
                    auxCoeff = net.at(layerNum).at(i).at(0);
                else
                    auxCoeff = 0;

                for ( auto k = 0; k < previousInfo.size(); k++ )
                    auxCoeff = auxCoeff + previousInfo.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

                auxNode.push_back(auxCoeff);
            }

            nextInfo.push_back(auxNode);
        }
    }

    BoundProtIndex firstBoundProtIdx = boundProts.size();

    for ( auto i = 0; i < nextInfo.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < nextInfo.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = nextInfo.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }

    if ( layerNum + 1 < net.size() )
    {
        vector<BoundProtPosition> positions;
        vector<unsigned> iteration;

        for ( auto i = firstBoundProtIdx; i < firstBoundProtIdx + nextInfo.size(); i++ )
        {
            BoundProtPosition auxBoundProtPos = boundProtPosition(i);
            positions.push_back( auxBoundProtPos );
        }

        vector<vector<Boundary>> futureBounds;
        vector<Boundary> dummyBounds;

        iterate_mthreading(futureBounds, currentBounds, dummyBounds, firstBoundProtIdx, positions, 0);

        for ( auto i = 0; i < futureBounds.size(); i++ )
        {
            vector<Node> auxNextInfo = nextInfo;
            unsigned fBoundsPos = 0;

            for ( auto j = 0; j < positions.size(); j++ )
            {
                if ( positions.at(j) == Cutting )
                {
                    if ( futureBounds.at(i).at(fBoundsPos).second == LeqZero )
                        for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                            auxNextInfo.at(j).at(k) = 0;

                    fBoundsPos++;
                }
                else if ( positions.at(j) == Under )
                {
                    for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                        auxNextInfo.at(j).at(k) = 0;
                }
            }

            vector<Boundary> auxCurrentBounds(currentBounds);
            auxCurrentBounds.insert(auxCurrentBounds.end(), futureBounds.at(i).begin(), futureBounds.at(i).end());

            net2pwl_mthreading(auxNextInfo, auxCurrentBounds, layerNum+1);
        }
    }
    else
    {
        RegionalLinearPiece rlp0;
        rlp0.bounds = currentBounds;
        rlp0.bounds.push_back(Boundary(firstBoundProtIdx, LeqZero));
        if ( feasibleBounds(rlp0.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp0);
        }

        boundProts.push_back(boundProts.at(firstBoundProtIdx));
        boundProts.at(firstBoundProtIdx+1).at(0) = boundProts.at(firstBoundProtIdx+1).at(0) - 1;

        RegionalLinearPiece rlp0_1;
        rlp0_1.bounds = currentBounds;
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx, GeqZero));
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx+1, LeqZero));
        if ( feasibleBounds(rlp0_1.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0_1.coefs.push_back(dec2frac(nextInfo.at(0).at(i)));
            rlPieces.push_back(rlp0_1);
        }

        RegionalLinearPiece rlp1;
        rlp1.bounds = currentBounds;
        rlp1.bounds.push_back(Boundary(firstBoundProtIdx+1, GeqZero));
        if ( feasibleBounds(rlp1.bounds) )
        {
            rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
            for ( auto i = 1; i < nextInfo.at(0).size(); i++ )
                rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp1);
        }
    }
}

void NeuralNetwork::net2pwl_singleThread(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<Node> nextInfo;

    if ( layerNum == 0 )
        nextInfo = previousInfo;
    else
    {
        for ( auto i = 0; i < net.at(layerNum).size(); i++ )
        {
            Node auxNode;

            for ( auto j = 0; j < previousInfo.at(0).size(); j++ )
            {
                NodeCoeff auxCoeff;

                if ( j == 0 )
                    auxCoeff = net.at(layerNum).at(i).at(0);
                else
                    auxCoeff = 0;

                for ( auto k = 0; k < previousInfo.size(); k++ )
                    auxCoeff = auxCoeff + previousInfo.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

                auxNode.push_back(auxCoeff);
            }

            nextInfo.push_back(auxNode);
        }
    }

    BoundProtIndex firstBoundProtIdx = boundProts.size();

    for ( auto i = 0; i < nextInfo.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < nextInfo.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = nextInfo.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }

    if ( layerNum + 1 < net.size() )
    {
        vector<BoundProtPosition> positions;
        vector<unsigned> iteration;

        for ( auto i = firstBoundProtIdx; i < firstBoundProtIdx + nextInfo.size(); i++ )
        {
            BoundProtPosition auxBoundProtPos = boundProtPosition(i);
            positions.push_back( auxBoundProtPos );
            iteration.push_back(0);
        }

        int iterationPosition = 0;
        vector<Boundary> auxCurrentBounds = currentBounds;

        while ( iterationPosition >= 0 )
        {
            vector<Node> auxNextInfo = nextInfo;

            while ( ( iterationPosition < iteration.size() ) && ( iterationPosition >= 0 ) )
            {
                if ( positions.at(iterationPosition) != Cutting )
                    iterationPosition++;
                else
                {
                    if ( iteration.at(iterationPosition) == 1 )
                        auxCurrentBounds.push_back( Boundary(firstBoundProtIdx+iterationPosition,GeqZero) );
                    else
                        auxCurrentBounds.push_back( Boundary(firstBoundProtIdx+iterationPosition,LeqZero) );

                    if ( feasibleBounds( auxCurrentBounds ) )
                        iterationPosition++;
                    else
                    {
                        bool iterating = true;
                        while ( iterating && ( iterationPosition >= 0 ) )
                        {
                            if ( positions.at(iterationPosition) == Cutting )
                                auxCurrentBounds.pop_back();

                            if ( ( positions.at(iterationPosition) == Cutting ) && ( iteration.at(iterationPosition) == 0 ) )
                            {
                                iteration.at(iterationPosition)++;
                                iterating = false;
                            }
                            else
                            {
                                iteration.at(iterationPosition) = 0;
                                iterationPosition--;
                            }
                        }
                    }
                }
            }

            if ( iterationPosition >= 0 )
            {
                for ( auto j = 0; j < iteration.size(); j++ )
                {
                    if ( positions.at(j) == Cutting )
                    {
                        if ( iteration.at(j) == 0 )
                            for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                                auxNextInfo.at(j).at(k) = 0;
                    }
                    else if ( positions.at(j) == Under )
                    {
                        for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                            auxNextInfo.at(j).at(k) = 0;
                    }
                }

                net2pwl_singleThread(auxNextInfo, auxCurrentBounds, layerNum+1);

                iterationPosition--;
                bool iterating = true;
                while ( iterating && iterationPosition >= 0 )
                {
                    if ( positions.at(iterationPosition) == Cutting )
                        auxCurrentBounds.pop_back();

                    if ( ( positions.at(iterationPosition) == Cutting ) && ( iteration.at(iterationPosition) == 0 ) )
                    {
                        iteration.at(iterationPosition)++;
                        iterating = false;
                    }
                    else
                    {
                        iteration.at(iterationPosition) = 0;
                        iterationPosition--;
                    }
                }
            }
        }
    }
    else
    {
        RegionalLinearPiece rlp0;
        rlp0.bounds = currentBounds;
        rlp0.bounds.push_back(Boundary(firstBoundProtIdx, LeqZero));
        if ( feasibleBounds(rlp0.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp0);
        }

        boundProts.push_back(boundProts.at(firstBoundProtIdx));
        boundProts.at(firstBoundProtIdx+1).at(0) = boundProts.at(firstBoundProtIdx+1).at(0) - 1;

        RegionalLinearPiece rlp0_1;
        rlp0_1.bounds = currentBounds;
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx, GeqZero));
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx+1, LeqZero));
        if ( feasibleBounds(rlp0_1.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0_1.coefs.push_back(dec2frac(nextInfo.at(0).at(i)));
            rlPieces.push_back(rlp0_1);
        }

        RegionalLinearPiece rlp1;
        rlp1.bounds = currentBounds;
        rlp1.bounds.push_back(Boundary(firstBoundProtIdx+1, GeqZero));
        if ( feasibleBounds(rlp1.bounds) )
        {
            rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
            for ( auto i = 1; i < nextInfo.at(0).size(); i++ )
                rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp1);
        }
    }
}

// old versions of net2pwl function
/*
void NeuralNetwork::net2pwl(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<Node> nextInfo;

    if ( layerNum == 0 )
        nextInfo = previousInfo;
    else
    {
        for ( auto i = 0; i < net.at(layerNum).size(); i++ )
        {
            Node auxNode;

            for ( auto j = 0; j < previousInfo.at(0).size(); j++ )
            {
                NodeCoeff auxCoeff;

                if ( j == 0 )
                    auxCoeff = net.at(layerNum).at(i).at(0);
                else
                    auxCoeff = 0;

                for ( auto k = 0; k < previousInfo.size(); k++ )
                    auxCoeff = auxCoeff + previousInfo.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

                auxNode.push_back(auxCoeff);
            }

            nextInfo.push_back(auxNode);
        }
    }

    BoundProtIndex firstBoundProtIdx = boundProts.size();

    for ( auto i = 0; i < nextInfo.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < nextInfo.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = nextInfo.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }

    if ( layerNum + 1 < net.size() )
    {
        vector<BoundProtPosition> positions;
        vector<unsigned> iteration;
        long nextInfoNum = 1;

        for ( auto i = firstBoundProtIdx; i < firstBoundProtIdx + nextInfo.size(); i++ )
        {
            BoundProtPosition auxBoundProtPos = boundProtPosition(i);
            positions.push_back( auxBoundProtPos );
            iteration.push_back(0);
            if ( auxBoundProtPos == Cutting )
                nextInfoNum = nextInfoNum * 2;
        }

        for ( auto i = 0; i < nextInfoNum; i++ )
        {
            vector<Node> auxNextInfo = nextInfo;
            vector<Boundary> auxCurrentBounds = currentBounds;

            for ( auto j = 0; j < iteration.size(); j++ )
            {
                Boundary auxBound;
                auxBound.first = firstBoundProtIdx + j;

                if ( positions.at(j) == Cutting )
                {
                    if ( iteration.at(j) == 1 )
                        auxBound.second = GeqZero;
                    else if ( iteration.at(j) == 0 )
                    {
                        auxBound.second = LeqZero;

                        for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                            auxNextInfo.at(j).at(k) = 0;
                    }

                    auxCurrentBounds.push_back(auxBound);
                }
                else if ( positions.at(j) == Under )
                {
                    for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                        auxNextInfo.at(j).at(k) = 0;
                }
                else if ( positions.at(j) == Over ) {}
            }

            if ( i + 1 != nextInfoNum )
            {
                bool iterated = false;

                for ( auto j = 0; ( (j < iteration.size()) && (!iterated) ); j++ )
                {
                    if ( ( positions.at(j) == Cutting ) && ( iteration.at(j) < 1 ) )
                    {
                        iteration.at(j)++;
                        iterated = true;
                    }
                    else
                        iteration.at(j) = 0;
                }
            }

            if ( feasibleBounds(auxCurrentBounds) )
                net2pwl(auxNextInfo, auxCurrentBounds, layerNum+1);
        }
    }
    else
    {
        RegionalLinearPiece rlp0;
        rlp0.bounds = currentBounds;
        rlp0.bounds.push_back(Boundary(firstBoundProtIdx, LeqZero));
        if ( feasibleBounds(rlp0.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp0);
        }

        boundProts.push_back(boundProts.at(firstBoundProtIdx));
        boundProts.at(firstBoundProtIdx+1).at(0) = boundProts.at(firstBoundProtIdx+1).at(0) - 1;

        RegionalLinearPiece rlp0_1;
        rlp0_1.bounds = currentBounds;
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx, GeqZero));
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx+1, LeqZero));
        if ( feasibleBounds(rlp0_1.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0_1.coefs.push_back(dec2frac(nextInfo.at(0).at(i)));
            rlPieces.push_back(rlp0_1);
        }

        RegionalLinearPiece rlp1;
        rlp1.bounds = currentBounds;
        rlp1.bounds.push_back(Boundary(firstBoundProtIdx+1, GeqZero));
        if ( feasibleBounds(rlp1.bounds) )
        {
            rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
            for ( auto i = 1; i < nextInfo.at(0).size(); i++ )
                rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp1);
        }
    }
}

void NeuralNetwork::net2pwl(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum)
{
    vector<Node> nextInfo;

    if ( layerNum == 0 )
        nextInfo = previousInfo;
    else
    {
        for ( auto i = 0; i < net.at(layerNum).size(); i++ )
        {
            Node auxNode;

            for ( auto j = 0; j < previousInfo.at(0).size(); j++ )
            {
                NodeCoeff auxCoeff;

                if ( j == 0 )
                    auxCoeff = net.at(layerNum).at(i).at(0);
                else
                    auxCoeff = 0;

                for ( auto k = 0; k < previousInfo.size(); k++ )
                    auxCoeff = auxCoeff + previousInfo.at(k).at(j) * net.at(layerNum).at(i).at(k+1);

                auxNode.push_back(auxCoeff);
            }

            nextInfo.push_back(auxNode);
        }
    }

    BoundProtIndex firstBoundProtIdx = boundProts.size();

    for ( auto i = 0; i < nextInfo.size(); i++ )
    {
        BoundaryPrototype auxBoundProt;

        for ( auto j = 0; j < nextInfo.at(0).size(); j++ )
        {
            BoundaryCoefficient auxCoeff;
            auxCoeff = nextInfo.at(i).at(j);
            auxBoundProt.push_back(auxCoeff);
        }

        boundProts.push_back(auxBoundProt);
    }

    if ( layerNum + 1 < net.size() )
    {
        for ( auto i = 0; i < pow(2, nextInfo.size()); i++ )
        {
            vector<Node> auxNextInfo = nextInfo;
            vector<Boundary> auxCurrentBounds = currentBounds;

            for ( auto j = 0; j < auxNextInfo.size(); j++ )
            {
                bool BoundProtConfig = (i >> j) & 1;

                Boundary auxBound;
                auxBound.first = firstBoundProtIdx + j;

                if ( BoundProtConfig )
                    auxBound.second = GeqZero;
                else
                {
                    auxBound.second = LeqZero;

                    for ( auto k = 0; k < auxNextInfo.at(0).size(); k++ )
                        auxNextInfo.at(j).at(k) = 0;
                }

                auxCurrentBounds.push_back(auxBound);
            }

            if ( feasibleBounds(auxCurrentBounds) )
                net2pwl(auxNextInfo, auxCurrentBounds, layerNum+1);
        }
    }
    else
    {
        RegionalLinearPiece rlp0;
        rlp0.bounds = currentBounds;
        rlp0.bounds.push_back(Boundary(firstBoundProtIdx, LeqZero));
        if ( feasibleBounds(rlp0.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp0);
        }

        boundProts.push_back(boundProts.at(firstBoundProtIdx));
        boundProts.at(firstBoundProtIdx+1).at(0) = boundProts.at(firstBoundProtIdx+1).at(0) - 1;

        RegionalLinearPiece rlp0_1;
        rlp0_1.bounds = currentBounds;
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx, GeqZero));
        rlp0_1.bounds.push_back(Boundary(firstBoundProtIdx+1, LeqZero));
        if ( feasibleBounds(rlp0_1.bounds) )
        {
            for ( auto i = 0; i < nextInfo.at(0).size(); i++ )
                rlp0_1.coefs.push_back(dec2frac(nextInfo.at(0).at(i)));
            rlPieces.push_back(rlp0_1);
        }

        RegionalLinearPiece rlp1;
        rlp1.bounds = currentBounds;
        rlp1.bounds.push_back(Boundary(firstBoundProtIdx+1, GeqZero));
        if ( feasibleBounds(rlp1.bounds) )
        {
            rlp1.coefs.push_back(LinearPieceCoefficient(1,1));
            for ( auto i = 1; i < nextInfo.at(0).size(); i++ )
                rlp1.coefs.push_back(LinearPieceCoefficient(0,1));
            rlPieces.push_back(rlp1);
        }
    }
}
*/

void NeuralNetwork::net2pwl()
{
    vector<Boundary> emptyBounds;

    if ( multithreading )
        net2pwl_mthreading(net.at(0), emptyBounds, 0);
    else
        net2pwl_singleThread(net.at(0), emptyBounds, 0);

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
{
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
    }
}
