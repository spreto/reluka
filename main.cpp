#include <iostream>
#include "Property.h"
#include "OnnxParser.h"
#include "NeuralNetwork.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"

bool printPwl = false;
bool printLimodsat = false;
bool printLiprop = false;
bool hasOnnx = false;
bool vnnlib = false;

std::string onnxFileName;
std::string vnnlibFileName;

void usage(std::string errorMessage)
{
    if ( !errorMessage.empty() )
        std::cout << "!!! " << errorMessage << std::endl << std::endl;

    std::cout << "Usage:" << std::endl;
    std::cout << "usage" << std::endl;
    std::cout << std::endl;
}

void usage()
{
    std::string emptyString;
    usage(emptyString);
}

void onlyIntermediateSteps()
{
    if ( !( printPwl || printLimodsat ) )
    {
        printPwl = true;
        printLimodsat = true;
    }

    reluka::OnnxParser onnx(onnxFileName);
    reluka::NeuralNetwork nn( onnx.getNeuralNetwork(), onnx.getOnnxFileName() );
    nn.buildPwlData();

    for ( size_t outIdx = 0; outIdx < nn.getOutputDimension(); outIdx++ )
    {
        if ( printPwl )
            nn.printPwlFile(outIdx);

        if ( printLimodsat )
        {
            pwl2limodsat::PiecewiseLinearFunction pwl( nn.getPwlData(outIdx),
                                                       nn.getBoundProtData(),
                                                       nn.getPwlFileName(outIdx) );
            pwl.printLimodsatFile();
        }
    }
}

void vnnlibRoutine()
{
    pwl2limodsat::VariableManager vm;
    reluka::Property vnnlib( vnnlibFileName, &vm );
    vnnlib.buildProperty();
    reluka::OnnxParser onnx( onnxFileName );
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), vnnlib.getNnOutputIndexes(), onnx.getOnnxFileName());
    nn.buildPwlData();

    for ( unsigned nnOutputIdx : nn.getNnOutputIndexes() )
    {
        if ( printPwl )
            nn.printPwlFile(nnOutputIdx);

        pwl2limodsat::PiecewiseLinearFunction pwl( nn.getPwlData(nnOutputIdx),
                                                   nn.getBoundProtData(),
                                                   nn.getPwlFileName(nnOutputIdx),
                                                   &vm );

        if ( printLimodsat )
            pwl.printLimodsatFile();
        else
            pwl.representModsat();

        pwl.equivalentTo(vnnlib.getVariable(nnOutputIdx));
        vnnlib.setOutputAddress(&pwl);
    }

    if ( printLiprop )
        vnnlib.printLipropFile();
}

int main(int argc, char **argv)
{
    for ( int argNum = 1; argNum < argc; argNum++ )
    {
        if ( argv[argNum] == "-pwl" )
            printPwl = true;
        else if ( argv[argNum] == "-limodsat" )
            printLimodsat = true;
        else if ( argv[argNum] == "-liprop" )
            printLiprop = true;
        else if ( argv[argNum] == "-print" )
        {
            printPwl = true;
            printLimodsat = true;
            printLiprop = true;
        }
        else if ( argv[argNum] == "-onnx" )
        {
            argNum++;
            if ( argv[argNum][0] == '-' )
                throw std::invalid_argument("Missing onnx file path.");
            onnxFileName = argv[argNum];
            hasOnnx = true;
        }
        else if ( argv[argNum] == "-vnnlib" )
        {
            argNum++;
            if ( argv[argNum][0] == '-' )
                throw std::invalid_argument("Missing vnnlib file path.");
            vnnlibFileName = argv[argNum];
            vnnlib = true;
        }
    }

    if ( !hasOnnx )
        usage("A onnx file must be provided");
    else if ( !vnnlib )
        onlyIntermediateSteps();
    else if ( vnnlib )
        vnnlibRoutine();
    else
        usage();

    return 0;
}
