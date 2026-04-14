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
        OnnxParser(std::string inputOnnxFileName, bool inputAcasxu);
        NeuralNetworkData getNeuralNetwork();
        std::string getOnnxFileName() { return onnxFileName; }
        size_t getInputDim();
        void normalizeInput( unsigned inputNum, double inputMin, double inputMax );

    private:
        onnx::ModelProto onnxNeuralNetwork;
        NeuralNetworkData neuralNetwork;
        bool acasxu = false;

        bool netTranslation = false;

        std::string onnxFileName;

        unsigned layerMulAddRelu(unsigned node);
        void onnx2netRegular();

        // methods for parsing ACAS Xu neural network
        void getWeights(unsigned layNum);
        void onnx2net4acasxu();

        void onnx2net();
};
}

#endif // ONNXPARSER_H
