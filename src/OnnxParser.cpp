#include <fstream>

#include "OnnxParser.h"

namespace reluka
{
OnnxParser::OnnxParser(std::string inputOnnxFileName) : onnxFileName(inputOnnxFileName)
{
    std::ifstream onnxFile(inputOnnxFileName, std::ios_base::binary);
    onnxNeuralNetwork.ParseFromIstream(&onnxFile);
    onnxFile.close();
}

unsigned OnnxParser::layerMulAddRelu(unsigned beginingNode)
{
/*    if ( ( onnxNeuralNetwork.graph().node(beginingNode+1).op_type().compare("Add") != 0 ) ||
         ( ( onnxNeuralNetwork.graph().node(beginingNode+2).op_type().compare("Relu") != 0 ) &&
           ( onnxNeuralNetwork.graph().node(beginingNode+2).op_type().compare("Clip") != 0 ) ) ||
         ( onnxNeuralNetwork.graph().node(beginingNode).output(0).compare( onnxNeuralNetwork.graph().node(beginingNode+1).input(1) ) != 0 ) ||
         ( onnxNeuralNetwork.graph().node(beginingNode+1).output(0).compare( onnxNeuralNetwork.graph().node(beginingNode+2).input(0) ) != 0 ) )
        throw std::invalid_argument("Not a recognizable onnx format."); */

    int biasIdx = 0;
//    while ( onnxNeuralNetwork.graph().initializer(biasIdx).name().compare( onnxNeuralNetwork.graph().node(beginingNode+1).input(1) ) != 0 )
    while ( onnxNeuralNetwork.graph().initializer(biasIdx).name().compare( onnxNeuralNetwork.graph().node(beginingNode+1).input(0) ) != 0 )
    {
        biasIdx++;

        if ( biasIdx == onnxNeuralNetwork.graph().initializer_size() )
            throw std::invalid_argument("Not a recognizable onnx format.");
    }

    int weightsIdx = 0;
    while ( onnxNeuralNetwork.graph().initializer(weightsIdx).name().compare( onnxNeuralNetwork.graph().node(beginingNode).input(1) ) != 0 )
    {
        weightsIdx++;

        if ( weightsIdx == onnxNeuralNetwork.graph().initializer_size() )
            throw std::invalid_argument("Not a recognizable onnx format.");
    }

    if ( ( onnxNeuralNetwork.graph().initializer(biasIdx).dims_size() != 1 ) ||
         ( onnxNeuralNetwork.graph().initializer(biasIdx).dims(0) != onnxNeuralNetwork.graph().initializer(weightsIdx).dims(1) ) )
        throw std::invalid_argument("Not a recognizable onnx format.");

    Layer lay;

    for ( auto i = 0; i < onnxNeuralNetwork.graph().initializer(weightsIdx).dims(1); i++ )
    {
        Node noh;
        NodeCoefficient aux;

        memcpy(&aux, &onnxNeuralNetwork.graph().initializer(biasIdx).raw_data()[4*i], sizeof(aux));
        noh.push_back(aux);

        for ( auto j = 0; j < onnxNeuralNetwork.graph().initializer(weightsIdx).dims(0); j++ )
        {
            memcpy(&aux,
                   &onnxNeuralNetwork.graph().initializer(weightsIdx).raw_data()[4*((onnxNeuralNetwork.graph().initializer(weightsIdx).dims(1)*j)+i)],
                   sizeof(aux));
            noh.push_back(aux);
        }

        lay.push_back(noh);
    }

    neuralNetwork.push_back(lay);

    // Return 1 if it should be the last node and 0 otherwise
    if ( onnxNeuralNetwork.graph().node(beginingNode+2).op_type().compare("Relu") != 0 )
        return 1;
    return 0;
}

void OnnxParser::onnx2net()
{
    int currentNode = 0;

    while ( currentNode < onnxNeuralNetwork.graph().node_size() &&
            ( onnxNeuralNetwork.graph().node(currentNode).op_type().compare("Flatten") == 0 ||
              onnxNeuralNetwork.graph().node(currentNode).op_type().compare("Sub") == 0 ) )
        currentNode++;

    if ( currentNode == onnxNeuralNetwork.graph().node_size() )
        throw std::invalid_argument("Not a recognizable onnx format.");

    while ( currentNode < onnxNeuralNetwork.graph().node_size() )
    {
        if ( onnxNeuralNetwork.graph().node(currentNode).op_type().compare("MatMul") == 0 )
        {
            if ( layerMulAddRelu(currentNode) && currentNode + 3 < onnxNeuralNetwork.graph().node_size() )
                throw std::invalid_argument("Not a recognizable onnx format.");
            currentNode = currentNode + 3;
        }
        else
        {
            throw std::invalid_argument("Not a recognizable onnx format.");
            currentNode++; // isso aqui faz sentido??
        }
    }

    netTranslation = true;
}

NeuralNetworkData OnnxParser::getNeuralNetwork()
{
    if ( !netTranslation )
        onnx2net();

    return neuralNetwork;
}

void OnnxParser::normalizeInput( unsigned inputNum, double inputMin, double inputMax )
{
    if ( !netTranslation )
        onnx2net();

    for ( Node node : neuralNetwork.at(0) )
    {
        node.at(0) += ( node.at(0)*inputMin ); std::cout << node.at(inputNum+1) << "  ";
        node.at(inputNum+1) *= ( inputMax-inputMin ); std::cout << node.at(inputNum+1) << std::endl;
    }
}

void OnnxParser::centralizeOutput( unsigned outputNum, double center )
{
    if ( !netTranslation )
        onnx2net();

    neuralNetwork.back().at(outputNum).at(0) += ( 0.5 - center );
}
}
