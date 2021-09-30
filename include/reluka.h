#ifndef RELUKA_H_INCLUDED
#define RELUKA_H_INCLUDED

#include <vector>

namespace reluka
{
typedef float NodeCoefficient;
typedef std::vector<NodeCoefficient> Node;
typedef std::vector<Node> Layer;
typedef std::vector<Layer> NeuralNetworkData;

enum BoundProtPosition { Under, Cutting, Over };
}

#endif // RELUKA_H_INCLUDED
