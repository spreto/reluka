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
        vector<Layer> net;

        vector<RegionalLinearPiece> rlPieces;
        vector<BoundaryPrototype> boundProts;
        bool multithreading;
        bool pwlTranslation = false;
        mutex multithreadingMutex;

        string pwlFileName;

        lpcNonNegative gcd(lpcNonNegative a, lpcNonNegative b);
        LinearPieceCoefficient dec2frac(const NodeCoeff& decValue);
        BoundProtPosition boundProtPosition(BoundProtIndex bIdx);
        bool feasibleBoundsWithExtra(const vector<Boundary>& bounds, const vector<Boundary>& extraBounds);
        bool feasibleBounds(const vector<Boundary>& bounds);
        void iterate_mthreading(vector<vector<Boundary>>& fBounds, const vector<Boundary>& cBounds, vector<Boundary> tBounds, BoundProtIndex fbpIdx, const vector<BoundProtPosition>& pos, unsigned posIdx);

        void net2pwl_mthreading(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum);
        void net2pwl_singleThread(const vector<Node>& previousInfo, const vector<Boundary>& currentBounds, unsigned layerNum);
        void net2pwl();
};

#endif // NEURALNETWORK_H
