#include <fstream>

#include "OnnxParser.h"

using namespace std;

OnnxParser::OnnxParser(const char* inputOnnxFileName) : onnxFileName(inputOnnxFileName)
{
    ifstream onnxFile(inputOnnxFileName, ios_base::binary);
    onnxNet.ParseFromIstream(&onnxFile);
    onnxFile.close();
}

unsigned OnnxParser::layerMulAddRelu(unsigned beginingNode)
{
    if ( ( onnxNet.graph().node(beginingNode+1).op_type().compare("Add") != 0 ) ||
         ( ( onnxNet.graph().node(beginingNode+2).op_type().compare("Relu") != 0 ) &&
           ( onnxNet.graph().node(beginingNode+2).op_type().compare("Clip") != 0 ) ) ||
         ( onnxNet.graph().node(beginingNode).output(0).compare( onnxNet.graph().node(beginingNode+1).input(1) ) != 0 ) ||
         ( onnxNet.graph().node(beginingNode+1).output(0).compare( onnxNet.graph().node(beginingNode+2).input(0) ) != 0 ) )
        cout << "EXXCESS" << endl;

    unsigned biasIdx = 0;
    while ( onnxNet.graph().initializer(biasIdx).name().compare( onnxNet.graph().node(beginingNode+1).input(0) ) != 0 )
    {
        biasIdx++;

        if ( biasIdx == onnxNet.graph().initializer_size() )
            cout << "EXCESSAOOO" << endl;
    }

    unsigned weightsIdx = 0;
    while ( onnxNet.graph().initializer(weightsIdx).name().compare( onnxNet.graph().node(beginingNode).input(1) ) != 0 )
    {
        weightsIdx++;

        if ( weightsIdx == onnxNet.graph().initializer_size() )
            cout << "EXCESSAOOO" << endl;
    }

    if ( ( onnxNet.graph().initializer(biasIdx).dims_size() != 1 ) ||
         ( onnxNet.graph().initializer(biasIdx).dims(0) != onnxNet.graph().initializer(weightsIdx).dims(1) ) )
        cout << "MAIS EXCESSAO" << endl;

    Layer lay;

    for ( auto i = 0; i < onnxNet.graph().initializer(weightsIdx).dims(1); i++ )
    {
        Node noh;
        NodeCoeff aux;

        memcpy(&aux, &onnxNet.graph().initializer(biasIdx).raw_data()[4*i], sizeof(aux));
        noh.push_back(aux);

        for ( auto j = 0; j < onnxNet.graph().initializer(weightsIdx).dims(0); j++ )
        {
            memcpy(&aux, &onnxNet.graph().initializer(weightsIdx).raw_data()[4*((onnxNet.graph().initializer(weightsIdx).dims(1)*j)+i)], sizeof(aux));
            noh.push_back(aux);
        }

        lay.push_back(noh);
    }

    net.push_back(lay);

    // Return 1 if it should be the last node and 0 otherwise
    if ( onnxNet.graph().node(beginingNode+2).op_type().compare("Relu") )
        return 1;
    return 0;
}

void OnnxParser::onnx2net()
{
    unsigned currentNode = 0;

    while ( currentNode < onnxNet.graph().node_size() )
    {
        if ( onnxNet.graph().node(currentNode).op_type().compare("MatMul") == 0 )
        {
            if ( layerMulAddRelu(currentNode) && currentNode + 3 < onnxNet.graph().node_size() )
                cout << "EXCESSAOOO!!!!" << endl;
            currentNode = currentNode + 3;
        }
        else
        {
            cout << "Exception" << endl;
            currentNode++;
        }
    }

    netTranslation = true;
}

vector<Layer> OnnxParser::getNet()
{
    if ( !netTranslation )
        onnx2net();

    return net;
}
