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

#ifndef PWL2LIMODSAT_H_INCLUDED
#define PWL2LIMODSAT_H_INCLUDED

#include <vector>

namespace pwl2limodsat
{
enum ExecutionMode { PWL, TL };

typedef int LPCoefInteger;
typedef unsigned LPCoefNonNegative;
typedef std::pair<LPCoefInteger,LPCoefNonNegative> LinearPieceCoefficient;
typedef std::vector<LinearPieceCoefficient> LinearPieceData;

typedef size_t BoundProtIndex;
enum BoundarySymbol { GeqZero, LeqZero };
typedef std::pair<BoundProtIndex,BoundarySymbol> Boundary;
typedef std::vector<Boundary> BoundaryCollection;

struct RegionalLinearPieceData
{
    LinearPieceData lpData;
    BoundaryCollection bound;
};
typedef std::vector<RegionalLinearPieceData> PiecewiseLinearFunctionData;

typedef double BoundaryCoefficient;
typedef std::vector<BoundaryCoefficient> BoundaryPrototype;
typedef std::vector<BoundaryPrototype> BoundaryPrototypeCollection;

typedef unsigned Variable;
}

#endif // PWL2LIMODSAT_H_INCLUDED
