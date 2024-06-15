#include <fstream>

#include "OnnxParser4ACASXu.h"

namespace reluka
{
OnnxParser4ACASXu::OnnxParser4ACASXu(std::string inputOnnxFileName) : onnxFileName(inputOnnxFileName)
{
    std::ifstream onnxFile(inputOnnxFileName, std::ios_base::binary);
    onnxNeuralNetwork.ParseFromIstream(&onnxFile);
    onnxFile.close();
}

void OnnxParser4ACASXu::getWeights(unsigned layNum)
{
    Layer lay;

    for ( auto i = 0; i < onnxNeuralNetwork.graph().initializer(layNum).dims(1); i++ )
    {
        Node noh;
        NodeCoefficient aux;

        memcpy(&aux, &onnxNeuralNetwork.graph().initializer(layNum+1).raw_data()[4*i], sizeof(aux));
        noh.push_back(aux);

        for ( auto j = 0; j < onnxNeuralNetwork.graph().initializer(layNum).dims(0); j++ )
        {
            memcpy(&aux,
                   &onnxNeuralNetwork.graph().initializer(layNum).raw_data()[4*((onnxNeuralNetwork.graph().initializer(layNum).dims(1)*j)+i)],
                   sizeof(aux));
            noh.push_back(aux);
        }

        lay.push_back(noh);
    }

    neuralNetwork.push_back(lay);
}

void OnnxParser4ACASXu::onnx2net()
{
    unsigned layNum = 0;

    while ( layNum < onnxNeuralNetwork.graph().initializer_size() )
    {
        if ( onnxNeuralNetwork.graph().initializer(layNum).name().back() == 'W' )
            getWeights(layNum);
        layNum++;
    }

    netTranslation = true;
}

NeuralNetworkData OnnxParser4ACASXu::getNeuralNetwork()
{
    if ( !netTranslation )
        onnx2net();

    return neuralNetwork;
}

void OnnxParser4ACASXu::normalizeInput( unsigned inputNum, double inputMin, double inputMax )
{
    if ( !netTranslation )
        onnx2net();

    for ( Node node : neuralNetwork.at(0) )
    {
        node.at(0) += ( node.at(inputNum)*inputMin );
        node.at(inputNum) *= ( inputMax-inputMin );
    }
}

void OnnxParser4ACASXu::centralizeOutput( unsigned outputNum, double center )
{
    if ( !netTranslation )
        onnx2net();

    neuralNetwork.back().at(outputNum).at(0) += ( 0.5 - center );
}
}
