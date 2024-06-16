# ReŁUkasiewicz

**ReŁUkasiewicz** extracts from a feedforward neural network with ReLU activation function in all neurons in its hidden layers and truncated identity function in all neurons in its output layer into the function computed by each output neuron in the pre-closed regional format.

## Theoretical Reference

Preto, S. and Finger, M. 2023. Effective reasoning over neural networks using Łukasiewicz logic. In Pascal Hitzler, Md Kamruzzaman Sarker & Aaron Eberhart, editors: *Compendium of Neurosymbolic Artificial Intelligence*, chapter 28, *Frontiers in Artificial Intelligence and Applications* 369, IOS Press, pp. 609–630.
[doi.org/10.3233/FAIA230160](https://doi.org/10.3233/FAIA230160)

## Installation

You must have in your computer the compilers **gcc** and **g++**, the **protobuf**, **pthread**, **GMP** and **zlib** libraries and the **SoPlex** callable library [(soplex.zib.de)](https://soplex.zib.de/).

To compile **pwl2limodsat**, all you have to do is type, at the root of the distribution directory:

> $ make

To remove all compiled files, just type:

> $ make clean

## Usage

Typing the following command at the root of the distribution directory, **ReŁUkasiewicz** takes as input file *neuralnet.onnx* and produces outputs *neuralnet_0.pwl*, ..., *neuralnet_N.pwl* as long as the *.onnx* file is in a supported format, where *N* is the number of output neurons:

> $ ./bin/Release/reluka -onnx neuralnet.onnx -pwl

Test scripts in folder *tests/* show how supported *.onnx* files are and how they might be generated using the *PyTorch library* for *Python*.

### Funding

This work was supported by grant #2021/03117-2, São Paulo Research Foundation (FAPESP).

This work was carried out at the Center for Artificial Intelligence (C4AI-USP), with support by the São Paulo Research Foundation (FAPESP), grant #2019/07665-4, and by the IBM Corporation.
