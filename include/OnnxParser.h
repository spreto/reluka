#ifndef ONNXPARSER_H
#define ONNXPARSER_H

#include <vector>

#include "onnx-ml.pb.h"

using namespace std;

typedef float NodeCoeff;
typedef vector<NodeCoeff> Node;
typedef vector<Node> Layer;

class OnnxParser
{
    public:
        OnnxParser(const char* inputOnnxFileName);
        vector<Layer> getNet();
        string getOnnxFileName() { return onnxFileName; }

    private:
        onnx::ModelProto onnxNet;
        vector<Layer> net;
        bool netTranslation = false;

        string onnxFileName;

        unsigned layerMulAddRelu(unsigned node);
        void onnx2net();
};

#endif // ONNXPARSER_H
