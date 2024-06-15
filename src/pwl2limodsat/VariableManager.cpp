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

#include <stdexcept>
#include "VariableManager.h"

namespace pwl2limodsat
{
VariableManager::VariableManager(unsigned dim) :
    counter(dim)
{
    counterInitialized = true;
}

VariableManager::VariableManager() {};

void VariableManager::setDimension(unsigned dim)
{
    if ( !counterInitialized )
    {
        counter = dim;
        counterInitialized = true;
    }
    else
        throw std::invalid_argument("Dimension already defined.");
}

void VariableManager::verifyInitialization()
{
    if ( !counterInitialized )
        throw std::invalid_argument("Variable Manager dimension still not set.");
}

void VariableManager::jumpToVariable(Variable toVar)
{
    verifyInitialization();

    if ( toVar >= counter )
        counter = toVar;
    else
        throw std::invalid_argument("Cannot jump to smaller variable number.");
}

unsigned VariableManager::currentVariable()
{
    verifyInitialization();

    return counter;
}

unsigned VariableManager::newVariable()
{
    verifyInitialization();

    counter++;

    return counter;
}

unsigned VariableManager::zeroVariable()
{
    verifyInitialization();

    if ( zero == 0 )
        zero = ++counter;

    return zero;
}

bool VariableManager::isThereConstant(LPCoefNonNegative denum)
{
    verifyInitialization();

    if ( constantsMap.find( denum ) != constantsMap.end() )
        return true;
    else
        return false;
}

Variable VariableManager::constant(LPCoefNonNegative denum)
{
    verifyInitialization();

    return constantsMap.find(denum)->second;
}

Variable VariableManager::newConstant(LPCoefNonNegative denum)
{
    verifyInitialization();

    constantsMap.insert(std::pair<LPCoefNonNegative,Variable>(denum,newVariable()));
    return currentVariable();
}

bool VariableManager::isThereAuxMultVariable(LPCoefNonNegative denum)
{
    verifyInitialization();

    if ( auxMultMap.find(denum) != auxMultMap.end() )
        return true;
    else
        return false;
}

Variable VariableManager::auxMultVariable(LPCoefNonNegative denum)
{
    verifyInitialization();

    if ( auxMultMap.find(denum) != auxMultMap.end() )
        return auxMultMap.find(denum)->second;
    else
    {
        auxMultMap.insert(std::pair<LPCoefNonNegative,Variable>(denum,newVariable()));
        return currentVariable();
    }
}
}
