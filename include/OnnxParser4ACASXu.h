#ifndef ONNXPARSER4ACASXU_H
#define ONNXPARSER4ACASXU_H

#include "reluka.h"
#include "onnx-ml.proto3.pb.h"

namespace reluka
{
class OnnxParser4ACASXu
{
    public:
        OnnxParser4ACASXu(std::string inputOnnxFileName);
        NeuralNetworkData getNeuralNetwork();
        std::string getOnnxFileName() { return onnxFileName; }
        void normalizeInput( unsigned inputNum, double inputMin, double inputMax );
        void centralizeOutput( unsigned outputNum, double center );

    private:
        onnx::ModelProto onnxNeuralNetwork;
        NeuralNetworkData neuralNetwork;
        bool netTranslation = false;

        std::string onnxFileName;

        void getWeights(unsigned layNum);
        void onnx2net();
};
}

#endif // ONNXPARSER4ACASXU_H
