#ifndef ONNXPARSER_H
#define ONNXPARSER_H

#include "reluka.h"
#include "onnx-ml.proto3.pb.h"

namespace reluka
{
class OnnxParser
{
    public:
        OnnxParser(std::string inputOnnxFileName);
        NeuralNetworkData getNeuralNetwork();
        std::string getOnnxFileName() { return onnxFileName; }
        void normalizeInput( unsigned inputNum, double inputMin, double inputMax );
        void centralizeOutput( unsigned outputNum, double center );

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
