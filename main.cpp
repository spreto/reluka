#include <iostream>
#include "OnnxParser.h"
#include "NeuralNetwork.h"

using namespace std;

int main(int argc, char **argv)
{
    OnnxParser in(argv[1]);
    NeuralNetwork nn(in.getNet(), in.getOnnxFileName());

    nn.printPwlFile();

    return 0;
}
