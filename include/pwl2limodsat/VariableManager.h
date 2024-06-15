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

#ifndef VARIABLEMANAGER_H
#define VARIABLEMANAGER_H

#include <map>
#include "pwl2limodsat.h"

namespace pwl2limodsat
{
class VariableManager
{
    public:
        VariableManager(unsigned dim);
        VariableManager();
        void setDimension(unsigned dim);
        void jumpToVariable(Variable toVar);
        Variable currentVariable();
        Variable newVariable();
        Variable zeroVariable();
        bool isThereConstant(LPCoefNonNegative denum);
        Variable constant(LPCoefNonNegative denum);
        Variable newConstant(LPCoefNonNegative denum);
        bool isThereAuxMultVariable(LPCoefNonNegative denum);
        Variable auxMultVariable(LPCoefNonNegative denum);

    protected:

    private:
        Variable counter;
        Variable zero = 0;
        bool zeroDefined = false;
        bool counterInitialized = false;
        std::map<LPCoefNonNegative,Variable> constantsMap;
        std::map<LPCoefNonNegative,Variable> auxMultMap;

        void verifyInitialization();
};
}

#endif // VARIABLEMANAGER_H
