#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include <mutex>

using namespace std;

typedef float NodeCoeff;
typedef vector<NodeCoeff> Node;
typedef vector<Node> Layer;

typedef int lpcInteger;
typedef unsigned lpcNonNegative;
typedef pair<lpcInteger,lpcNonNegative> LinearPieceCoefficient;

typedef unsigned BoundProtIndex;
enum BoundarySymbol { GeqZero, LeqZero };
typedef pair<BoundProtIndex,BoundarySymbol> Boundary;

struct RegionalLinearPiece
{
    vector<LinearPieceCoefficient> coefs;
    vector<Boundary> bounds;
};

typedef double BoundaryCoefficient;
typedef vector<BoundaryCoefficient> BoundaryPrototype;

enum BoundProtPosition { Under, Cutting, Over };

class NeuralNetwork
{
    public:
        NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName, bool mth);
        NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName);
        vector<RegionalLinearPiece> getRegLinPieces();
        vector<BoundaryPrototype> getBoundPrototypes();
        void printPwlFile();

    private:
        string pwlFileName;

        vector<Layer> net;

        vector<RegionalLinearPiece> rlPieces;
        vector<BoundaryPrototype> boundProts;
        bool pwlTranslation = false;

        mutex multithreadingMutex; // A SER EXCLUÍDO!
        mutex rlPiecesMutex;
        mutex boundProtsMutex;
        bool multithreading;

        lpcNonNegative gcd(lpcNonNegative a, lpcNonNegative b);
        LinearPieceCoefficient dec2frac(const NodeCoeff& decValue);

        BoundProtPosition boundProtPosition(BoundProtIndex bIdx);
        bool feasibleBounds(const vector<Boundary>& bounds);
        bool feasibleBoundsWithExtra(const vector<Boundary>& bounds, const vector<Boundary>& extraBounds);

        vector<BoundaryPrototype> composeBoundProts(const vector<BoundaryPrototype>& inputValues, unsigned layerNum);
        void writeBoundProts(const vector<BoundaryPrototype>& newBoundProts);
        void writeRegionalLinearPieces(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, BoundProtIndex newBoundProtFirstIdx);

        void iterate(vector<unsigned>& iteration, int& currentIterationIdx, const vector<BoundProtPosition>& boundProtPositions, vector<Boundary>& bounds);
        void net2pwl(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, unsigned layerNum);

        void iterate_mthreading(const vector<BoundaryPrototype>& newBoundProts, const vector<Boundary>& currentBounds, vector<Boundary> tBounds, BoundProtIndex fbpIdx, const vector<BoundProtPosition>& boundProtPositions, unsigned posIdx, unsigned layerNum);
        void net2pwl_mthreading(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, unsigned layerNum);

        void net2pwl();
};

#endif // NEURALNETWORK_H
