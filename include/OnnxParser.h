#ifndef ONNXPARSER_H
#define ONNXPARSER_H

#include "reluka.h"
#include "onnx-ml.proto3.pb.h"

namespace reluka
{
class OnnxParser
{
    public:
        OnnxParser(const char* inputOnnxFileName);
        NeuralNetworkData getNeuralNetwork();
        std::string getOnnxFileName() { return onnxFileName; }

    private:
        onnx::ModelProto onnxNeuralNetwork;
        NeuralNetworkData neuralNetwork;
        bool netTranslation = false;

        std::string onnxFileName;

        unsigned layerMulAddRelu(unsigned node);
        void onnx2net();
};
}

#endif // ONNXPARSER_H
