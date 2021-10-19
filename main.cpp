#include <iostream>
#include "OnnxParser.h"
#include "NeuralNetwork.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"
#include "VnnlibProperty.h"
#include "GlobalRobustness.h"

bool printPwl = false;
bool printLimodsat = false;
bool printLiprop = false;
bool hasOnnx = false;
bool robust = false;
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
    reluka::VnnlibProperty vnnlib( vnnlibFileName, &vm );
    vnnlib.buildVnnlibProperty();
    reluka::OnnxParser onnx( onnxFileName );
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), vnnlib.getNnOutputIndexes(), onnx.getOnnxFileName());
    nn.buildPwlData();

    std::vector<pwl2limodsat::PiecewiseLinearFunction> pwl;

    for ( unsigned nnOutputIdx : nn.getNnOutputIndexes() )
    {
        if ( printPwl )
            nn.printPwlFile( nnOutputIdx );

        pwl2limodsat::PiecewiseLinearFunction pwlAux( nn.getPwlData(nnOutputIdx),
                                                      nn.getBoundProtData(),
                                                      nn.getPwlFileName(nnOutputIdx),
                                                      &vm );
        pwl.push_back( pwlAux );

        if ( printLimodsat )
            pwl.back().printLimodsatFile();
        else
            pwl.back().representModsat();
    }

    vnnlib.setOutputAddresses( &pwl );

    if ( printLiprop )
        vnnlib.printLipropFile();
}

void globalRobustnessRoutine()
{
    reluka::OnnxParser onnx( onnxFileName );
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), onnx.getOnnxFileName());
    nn.buildPwlData();
    pwl2limodsat::VariableManager vm( nn.getInputDimension() );

    std::vector<pwl2limodsat::PiecewiseLinearFunction> pwl;

    for ( unsigned nnOutputIdx : nn.getNnOutputIndexes() )
    {
        if ( printPwl )
            nn.printPwlFile( nnOutputIdx );

        pwl2limodsat::PiecewiseLinearFunction pwlAux( nn.getPwlData(nnOutputIdx),
                                                      nn.getBoundProtData(),
                                                      nn.getPwlFileName(nnOutputIdx),
                                                      &vm );
        pwl.push_back( pwlAux );

        if ( printLimodsat )
            pwl.back().printLimodsatFile();
        else
            pwl.back().representModsat();
    }

    reluka::GlobalRobustness globalRobust(onnx.getOnnxFileName(),
                                          nn.getInputDimension(),
                                          nn.getOutputDimension(),
                                          &pwl,
                                          0.5,
                                          &vm);

    if ( printLiprop )
        globalRobust.printLipropFile();
}

int main(int argc, char **argv)
{
    for ( int argNum = 1; argNum < argc; argNum++ )
    {
        std::string arg(argv[argNum]);

        if ( arg.compare("-pwl") == 0 )
            printPwl = true;
        else if ( arg.compare("-limodsat") == 0 )
            printLimodsat = true;
        else if ( arg.compare("-liprop") == 0 )
            printLiprop = true;
        else if ( arg.compare("-print") == 0 )
        {
            printPwl = true;
            printLimodsat = true;
            printLiprop = true;
        }
        else if ( arg.compare("-robust") == 0 )
            robust = true;
        else if ( arg.compare("-vnnlib") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing vnnlib file path.");
            vnnlibFileName = arg;
            vnnlib = true;
        }
        else if ( arg.compare("-onnx") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing onnx file path.");
            onnxFileName = arg;
            hasOnnx = true;
        }
    }

    if ( !hasOnnx )
        usage("A onnx file must be provided");
    else if ( !vnnlib && !robust )
        onlyIntermediateSteps();
    else if ( robust )
        globalRobustnessRoutine();
    else if ( vnnlib )
        vnnlibRoutine();
    else
        usage();

    return 0;
}
