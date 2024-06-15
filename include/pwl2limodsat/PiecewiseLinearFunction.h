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

#ifndef PIECEWISELINEARFUNCTION_H
#define PIECEWISELINEARFUNCTION_H

#include "RegionalLinearPiece.h"

namespace pwl2limodsat
{
class PiecewiseLinearFunction
{
    public:
        PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                const BoundaryPrototypeCollection& boundProtData,
                                std::string inputFileName,
                                VariableManager *varMan,
                                bool multithreading);
        PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                const BoundaryPrototypeCollection& boundProtData,
                                std::string inputFileName,
                                bool multithreading);
        PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                const BoundaryPrototypeCollection& boundProtData,
                                std::string inputFileName,
                                VariableManager *varMan);
        PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                const BoundaryPrototypeCollection& boundProtData,
                                std::string inputFileName);
        ~PiecewiseLinearFunction();
        bool hasLatticeProperty();
        unsigned long long int latticePropertyCounter();
        void representModsat();
        void equivalentTo(Variable variable);
        std::vector<RegionalLinearPiece> getLinearPieceCollection();
        Formula getLatticeFormula();
        void printLimodsatFile();

    protected:

    private:
        std::string outputFileName;
        enum ProcessingMode { Single, Multi };
        ProcessingMode processingMode;

        std::vector<RegionalLinearPiece> linearPieceCollection;
        BoundaryPrototypeCollection boundaryPrototypeData;
        Formula latticeFormula;

        VariableManager *var;
        bool ownVariableManager = false;
        bool modsatTranslation = false;

        void setProcessingMode(ProcessingMode mode) { processingMode = mode; }
        void representPiecesModSat();
        std::vector<Formula> partialPhiOmega(unsigned thread, unsigned compByThread);
        void representLatticeFormula(unsigned maxThreadsNum);
};
}

#endif // PIECEWISELINEARFUNCTION_H
