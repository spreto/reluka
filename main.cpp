#include <iostream>
#include "Property.h"
#include "OnnxParser.h"
#include "NeuralNetwork.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"

using namespace std;

int main(int argc, char **argv)
{

    reluka::OnnxParser onnx(argv[1]);
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), onnx.getOnnxFileName());
//    std::vector<unsigned> outIdx {1, 3};
//    outIdx.push_back(stoi(argv[2]));
//    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), outIdx, onnx.getOnnxFileName());
    for ( unsigned outIdx : nn.getNnOutputIndexes() )
    {
        nn.printPwlFile(outIdx);
        pwl2limodsat::PiecewiseLinearFunction pwl(nn.getPwlData(outIdx), nn.getBoundProtData(), nn.getPwlFileName(outIdx));
        pwl.printLimodsatFile();
    }

/*
    if ( pwl.hasLatticeProperty() )
        cout << "HAS the lattice property" << endl;
    else
        cout << "DOES NOT HAVE the lattice property" << endl;
*/
/*
    pwl2limodsat::VariableManager vm;
    reluka::Property vnnlib(argv[1], &vm);
    vnnlib.buildProperty();
    reluka::OnnxParser onnx(argv[2]);
    reluka::NeuralNetwork nn(onnx.getNeuralNetwork(), onnx.getOnnxFileName());
    pwl2limodsat::PiecewiseLinearFunction pwl(nn.getPwlData(0), nn.getBoundProtData(), nn.getPwlFileName(0), &vm);
//    pwl.equivalentTo(vnnlib.get);
    vnnlib.setOutputAddress(&pwl);
    vnnlib.printLipropFile();
*/
    return 0;
}
