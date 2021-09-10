#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

using namespace std;

enum ProcessingMode { Single, Multi };

typedef float NodeCoeff;
typedef vector<NodeCoeff> Node;
typedef vector<Node> Layer;
typedef vector<Layer> NeuralNetworkCircuit;

typedef int lpcInteger;
typedef unsigned lpcNonNegative;
typedef pair<lpcInteger,lpcNonNegative> LinearPieceCoefficient;

typedef size_t BoundProtIndex;
enum BoundarySymbol { GeqZero, LeqZero };
typedef pair<BoundProtIndex,BoundarySymbol> Boundary;

struct RegionalLinearPiece
{
    vector<LinearPieceCoefficient> coefs;
    vector<Boundary> bounds;
};
typedef vector<RegionalLinearPiece> PwlRegionalLinearPieces;

typedef double BoundaryCoefficient;
typedef vector<BoundaryCoefficient> BoundaryPrototype;
typedef vector<BoundaryPrototype> PwlBoundaryPrototypes;

typedef pair<PwlRegionalLinearPieces,PwlBoundaryPrototypes> PwlPartialInfo;

enum BoundProtPosition { Under, Cutting, Over };

class NeuralNetwork
{
    public:
        NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName, bool multithreading);
        NeuralNetwork(const vector<Layer>& inputNet, string onnxFileName);
        void setProcessingMode(ProcessingMode mode) { processingMode = mode; }
        PwlRegionalLinearPieces getRegLinPieces();
        PwlBoundaryPrototypes getBoundPrototypes();
        void printPwlFile();

    private:
        string pwlFileName;
        ProcessingMode processingMode;

        NeuralNetworkCircuit net;

        PwlRegionalLinearPieces rlPieces;
        PwlBoundaryPrototypes boundProts;

        bool pwlTranslation = false;

        lpcNonNegative gcd(lpcNonNegative a, lpcNonNegative b);
        LinearPieceCoefficient dec2frac(NodeCoeff decValue);

        BoundProtPosition boundProtPosition(const vector<BoundaryPrototype>& boundProts, BoundProtIndex bIdx);
        bool feasibleBounds(const vector<BoundaryPrototype>& boundProts, const vector<Boundary>& bounds);
        vector<BoundaryPrototype> composeBoundProts(const vector<BoundaryPrototype>& inputValues, unsigned layerNum);
        void writeBoundProts(vector<BoundaryPrototype>& boundProts, const vector<BoundaryPrototype>& newBoundProts);
        vector<BoundaryPrototype> composeOutputValues(const vector<BoundaryPrototype>& boundProts,
                                                      const vector<BoundarySymbol>& iteration,
                                                      const vector<BoundProtPosition>& boundProtPositions);
        void writeRegionalLinearPieces(vector<RegionalLinearPiece>& rlPieces,
                                       const vector<BoundaryPrototype>& boundProts,
                                       const vector<BoundaryPrototype>& inputValues,
                                       const vector<Boundary>& currentBounds,
                                       BoundProtIndex newBoundProtFirstIdx);
        bool iterate(size_t limitIterationIdx,
                     vector<BoundarySymbol>& iteration,
                     size_t& currentIterationIdx,
                     const vector<BoundProtPosition>& boundProtPositions,
                     vector<Boundary>& bounds);
        bool iterate(vector<BoundarySymbol>& iteration,
                     size_t& currentIterationIdx,
                     const vector<BoundProtPosition>& boundProtPositions,
                     vector<Boundary>& bounds);

        void net2pwl(vector<BoundaryPrototype>& boundProts,
                     vector<RegionalLinearPiece>& rlPieces,
                     const vector<BoundaryPrototype>& inputValues,
                     const vector<Boundary>& currentBounds,
                     size_t layerNum);
        void net2pwl(const vector<BoundaryPrototype>& inputValues, const vector<Boundary>& currentBounds, size_t layerNum);

        PwlPartialInfo partialNet2pwl(unsigned startingPoint,
                                      size_t fixedNodesNum,
                                      const vector<BoundaryPrototype>& inputValues,
                                      const vector<BoundProtPosition>& boundProtPositions);
        void pwlInfoMerge(const vector<PwlPartialInfo>& threadsInfo);
        void net2pwlMultithreading(const vector<BoundaryPrototype>& inputValues);

        void net2pwl();
};

#endif // NEURALNETWORK_H
