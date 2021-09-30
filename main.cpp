#include <iostream>
#include "OnnxParser.h"
#include "NeuralNetwork.h"
#include "PiecewiseLinearFunction.h"

using namespace std;

int main(int argc, char **argv)
{
    reluka::OnnxParser onnx(argv[1]);
    reluka::NeuralNetwork nn(onnx.getNet(), onnx.getOnnxFileName());
    nn.printPwlFile();

    pwl2limodsat::PiecewiseLinearFunction pwl(nn.getPwlData(), nn.getBoundProtData(), nn.getPwlFileName());
    pwl.printRepresentation(); // talvez deva ser printLiModSatFile...

    if ( pwl.hasLatticeProperty() )
        cout << "HAS the lattice property" << endl;
    else
        cout << "DOES NOT HAVE the lattice property" << endl;

    return 0;
}
