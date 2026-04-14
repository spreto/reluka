#include <iostream>
#include "OnnxParser.h"
#include "NeuralNetwork.h"
#include "ZhangBolcskeiModSat.h"
#include "NeuralNetworkModSat.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"
#include "InequalityConstraints.h"
#include "InequalitySatisfiability.h"
#include "VnnlibProperty.h"
#include "GlobalRobustness.h"

bool pwl = false;
bool verifyLatticeProperty = true;
bool latticePropertyCounter = false;
bool limodsat = false;
bool zblimodsat = false;
bool hasOnnx = false;
bool ineqcons = false;
bool ineqsat = false;
bool robust = false;
bool vnnlib = false;
bool acasxu = false;

std::string onnxFileName;
std::string ineqconsFileName;
std::string ineqsatFileName;
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
    reluka::OnnxParser onnx( onnxFileName, acasxu );

//    std::cout << "==WARNING: The neural network will not be normalized==" << std::endl;
//    std::cout << "The input must be a rational McNaughton neural network" << std::endl << std::endl;

    if ( pwl )
    {
        reluka::NeuralNetwork nn( onnx.getNeuralNetwork(), onnx.getOnnxFileName() );
        nn.buildPwlData();

        for ( size_t outIdx = 0; outIdx < nn.getOutputDimension(); outIdx++ )
        {
            nn.printPwlFile(outIdx);

            if ( limodsat || verifyLatticeProperty || latticePropertyCounter )
            {
                pwl2limodsat::PiecewiseLinearFunction pwl( nn.getPwlData((unsigned) outIdx),
                                                           nn.getBoundProtData(),
                                                           nn.getPwlFileName((unsigned) outIdx) );

                if ( verifyLatticeProperty && !pwl.hasLatticeProperty() )
                    throw std::domain_error("Pre-regional format without the lattice property.");
                else if ( latticePropertyCounter )
                    std::cout << "out" << outIdx << ": " << pwl.latticePropertyCounter() << std::endl;

                if ( limodsat )
                    pwl.printLimodsatFile();
            }
        }
    }
    else if ( zblimodsat )
    {
        reluka::ZhangBolcskeiModSat zbms( onnx.getNeuralNetwork(), onnx.getOnnxFileName() );

        for ( size_t outIdx = 0; outIdx < zbms.getOutputDimension(); outIdx++ )
            zbms.printZBmodsatFile((unsigned) outIdx);
    }
    else
    {
        reluka::NeuralNetworkModSat nnms( onnx.getNeuralNetwork(), onnx.getOnnxFileName() );

        for ( size_t outIdx = 0; outIdx < nnms.getOutputDimension(); outIdx++ )
            nnms.printNNmodsatFile((unsigned) outIdx);
    }
}

void inequalityConstraintsRoutine()
{
    reluka::OnnxParser onnx( onnxFileName, acasxu );
    pwl2limodsat::VariableManager vm;
    reluka::InequalityConstraints ineqcons( ineqconsFileName, onnx.getInputDim(), &vm );

    std::map<unsigned,std::pair<double,double>> inputLimits = ineqcons.getInputLimits();
    for ( auto& lim : inputLimits )
        onnx.normalizeInput(lim.first, lim.second.first, lim.second.second);

    reluka::NeuralNetworkModSat nnms( onnx.getNeuralNetwork(), ineqcons.getNnOutputIndexes(), onnx.getOnnxFileName(), true );
    ineqcons.buildIneqconsProperty( nnms.getOriginalOutputLim() );
    ineqcons.printLiproperty( &nnms );
}

void inequalitySatisfiabilityRoutine()
{
    reluka::OnnxParser onnx( onnxFileName, acasxu );
    pwl2limodsat::VariableManager vm;
    reluka::InequalitySatisfiability ineqsat( ineqsatFileName, onnx.getInputDim(), &vm );

    std::map<unsigned,std::pair<double,double>> inputLimits = ineqsat.getInputLimits();
    for ( auto& lim : inputLimits )
        onnx.normalizeInput(lim.first, lim.second.first, lim.second.second);

    reluka::NeuralNetworkModSat nnms( onnx.getNeuralNetwork(), ineqsat.getNnOutputIndexes(), onnx.getOnnxFileName(), true );
    ineqsat.buildIneqsatProperty( nnms.getOriginalOutputLim() );
    ineqsat.printLiproperty( &nnms );
}

/*
void globalRobustnessRoutine()
{
    reluka::OnnxParser onnx( onnxFileName );
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), onnx.getOnnxFileName());
    nn.buildPwlData();
    pwl2limodsat::VariableManager vm( nn.getInputDimension() );

    std::vector<pwl2limodsat::PiecewiseLinearFunction> pwl;

    for ( unsigned nnOutputIdx : nn.getNnOutputIndexes() )
    {
        if ( usePrintPwl )
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
        if ( usePrintPwl )
            nn.printPwlFile( nnOutputIdx );

        pwl2limodsat::PiecewiseLinearFunction pwlAux( nn.getPwlData(nnOutputIdx),
                                                      nn.getBoundProtData(),
                                                      nn.getPwlFileName(nnOutputIdx),
                                                      &vm );

        if ( verifyLatticeProperty && !pwlAux.hasLatticeProperty() )
            throw std::domain_error("Pre-regional format without the lattice property.");

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
*/

int main(int argc, char **argv)
{
    for ( int argNum = 1; argNum < argc; argNum++ )
    {
        std::string arg(argv[argNum]);

        if ( arg.compare("-pwl") == 0 )
            pwl = true;
        else if ( arg.compare("-without-lp") == 0 )
            verifyLatticeProperty = false;
        else if ( arg.compare("-lpcount") == 0 )
        {
            verifyLatticeProperty = false;
            latticePropertyCounter = true;
        }
        else if ( arg.compare("-limodsat") == 0 )
            limodsat = true;
        else if ( arg.compare("-zblimodsat") == 0 )
            zblimodsat = true;
        else if ( arg.compare("-ineqcons") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing inequality constraints file path.");
            ineqconsFileName = arg;
            ineqcons = true;
        }
        else if ( arg.compare("-ineqsat") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing inequality satisfiability file path.");
            ineqsatFileName = arg;
            ineqsat = true;
        }
        else if ( arg.compare("-vnnlib") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing vnnlib file path.");
            vnnlibFileName = arg;
            vnnlib = true;
        }
        else if ( arg.compare("-robust") == 0 )
            robust = true;
        else if ( arg.compare("-onnx") == 0 )
        {
            argNum++;
            arg = argv[argNum];
            if ( arg.compare(0, 1, "-") == 0 )
                throw std::invalid_argument("Missing onnx file path.");
            onnxFileName = arg;
            hasOnnx = true;
        }
        else if ( arg.compare("-acasxu") == 0 )
            acasxu = true;
    }

    if ( !hasOnnx )
        usage("A onnx file must be provided");
    else if ( !ineqcons && !ineqsat && !robust && !vnnlib )
        onlyIntermediateSteps();
    else if ( ineqcons )
        inequalityConstraintsRoutine();
    else if ( ineqsat )
        inequalitySatisfiabilityRoutine();
    else if ( robust )
{}//        globalRobustnessRoutine();
    else if ( vnnlib )
{}//        vnnlibRoutine();
    else
        usage();

    return 0;
}
