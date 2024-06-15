/*
    The code in this file may be found in
    http://github.com/spreto/pwl2limodsat
    and is available under the following license.

    MIT License

    Copyright (c) 2021 Sandro Preto

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef LINEARPIECE_H
#define LINEARPIECE_H

#include <vector>
#include <string>
#include <fstream>
#include "VariableManager.h"
#include "Formula.h"

using namespace lukaFormula;

namespace pwl2limodsat
{
class LinearPiece
{
    public:
        LinearPiece(const LinearPieceData& data, VariableManager *varMan);
        LinearPiece(const LinearPieceData& data, std::string inputFileName);
        ~LinearPiece();

        void representModsat();
        Modsat getRepresentationModsat();
        void printModsatSetAs(std::ofstream *output, std::string intro);
        void printModsatSet(std::ofstream *output);
        void printLimodsatFile();

        static Formula zeroFormula(VariableManager *var);
        template<class T> static Modsat binaryModsat(VariableManager *var, unsigned n, const T& logTerm);
        static ModsatSet defineConstant(VariableManager *var, unsigned denum);
        static Modsat multiplyConstant(VariableManager *var, unsigned num, unsigned denum);

    protected:
        LinearPieceData linearPieceData;
        unsigned dim;

    private:
        std::string outputFileName;

        Modsat representationModsat;

        VariableManager *var;
        bool ownVariableManager = false;
        bool modsatTranslation = false;

        Formula zeroFormula();
        template<class T> Modsat binaryModsat(unsigned n, const T& logTerm);
        ModsatSet defineConstant(unsigned denum);
        Modsat multiplyConstant(unsigned num, unsigned denum);
        Formula variableSecondMultiplication(unsigned n, Variable var);
        void pwl2limodsat();
};
}

#endif // LINEARPIECE_H
