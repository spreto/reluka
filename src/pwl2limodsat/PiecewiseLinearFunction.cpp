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

#include "PiecewiseLinearFunction.h"
#include <iostream>
#include <future>
#include <cmath>

namespace pwl2limodsat
{
PiecewiseLinearFunction::PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                                 const BoundaryPrototypeCollection& boundProtData,
                                                 std::string inputFileName,
                                                 VariableManager *varMan,
                                                 bool multithreading) :
    boundaryPrototypeData(boundProtData),
    var(varMan)
{
    for ( size_t i = 0; i < pwlData.size(); i++ )
        linearPieceCollection.push_back(RegionalLinearPiece(pwlData.at(i), &boundaryPrototypeData, varMan));

    if ( inputFileName.substr(inputFileName.size()-4,4) == ".pwl" )
        outputFileName = inputFileName.substr(0,inputFileName.size()-4);
    else
        outputFileName = inputFileName;

    processingMode = ( multithreading ? Multi : Single );

    outputFileName.append(".limodsat");
}

PiecewiseLinearFunction::PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                                 const BoundaryPrototypeCollection& boundProtData,
                                                 std::string inputFileName,
                                                 bool multithreading) :
    PiecewiseLinearFunction(pwlData,
                            boundProtData,
                            inputFileName,
                            new VariableManager(pwlData.at(0).lpData.size() - 1),
                            multithreading)
{
    ownVariableManager = true;
}

PiecewiseLinearFunction::PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                                 const BoundaryPrototypeCollection& boundProtData,
                                                 std::string inputFileName,
                                                 VariableManager *varMan) :
    PiecewiseLinearFunction(pwlData, boundProtData, inputFileName, varMan, false) {}

PiecewiseLinearFunction::PiecewiseLinearFunction(const PiecewiseLinearFunctionData& pwlData,
                                                 const BoundaryPrototypeCollection& boundProtData,
                                                 std::string inputFileName) :
    PiecewiseLinearFunction(pwlData, boundProtData, inputFileName, false) {}

PiecewiseLinearFunction::~PiecewiseLinearFunction()
{
    if ( ownVariableManager )
        delete var;
}

bool PiecewiseLinearFunction::hasLatticeProperty()
{
    bool hasLatticeProperty = true;

    for ( size_t i = 0; i < linearPieceCollection.size() && hasLatticeProperty; i++ )
        for ( size_t j = 0; j < linearPieceCollection.size() && hasLatticeProperty; j++ )
            if ( i != j )
            {
                bool found = false;

                for ( size_t k = 0; k < linearPieceCollection.size() && !found; k++ )
                {
                    if ( linearPieceCollection.at(i).comparedIsBelow(linearPieceCollection.at(k)) )
                        if ( linearPieceCollection.at(j).comparedIsAbove(linearPieceCollection.at(k)) )
                            found = true;
                }

                if ( !found )
                    hasLatticeProperty = false;
            }

    return hasLatticeProperty;
}

unsigned long long int PiecewiseLinearFunction::latticePropertyCounter()
{
    unsigned long long int counter = 0;

    for ( size_t i = 0; i < linearPieceCollection.size(); i++ )
        for ( size_t j = 0; j < linearPieceCollection.size(); j++ )
            if ( i != j )
            {
                bool found = false;

                for ( size_t k = 0; k < linearPieceCollection.size() && !found; k++ )
                {
                    if ( linearPieceCollection.at(i).comparedIsBelow(linearPieceCollection.at(k)) )
                        if ( linearPieceCollection.at(j).comparedIsAbove(linearPieceCollection.at(k)) )
                            found = true;
                }

                if ( !found )
                    counter++;
            }

    return counter;
}

void PiecewiseLinearFunction::representPiecesModSat()
{
    for ( size_t i = 0; i < linearPieceCollection.size(); i++ )
        linearPieceCollection.at(i).representModsat();
}

std::vector<Formula> PiecewiseLinearFunction::partialPhiOmega(unsigned thread, unsigned compByThread)
{
    std::vector<Formula> partPhiOmega;

    for ( size_t i = thread * compByThread; i < (thread + 1) * compByThread; i++ )
    {
        partPhiOmega.push_back( linearPieceCollection.at(i).getRepresentationModsat().phi );

        for ( size_t k = 0; k < linearPieceCollection.size(); k++ )
            if ( k != i )
                if ( linearPieceCollection.at(i).comparedIsAbove(linearPieceCollection.at(k)) )
                    partPhiOmega.back().addMinimum(linearPieceCollection.at(k).getRepresentationModsat().phi);
    }

    return partPhiOmega;
}

void PiecewiseLinearFunction::representLatticeFormula(unsigned maxThreadsNum)
{
    unsigned compByThread = ceil( (float) linearPieceCollection.size() / (float) maxThreadsNum );
    unsigned threadsNum = ceil( (float) linearPieceCollection.size() / (float) compByThread );

    std::vector<std::future<std::vector<Formula>>> phiOmegaFut;
    for ( unsigned thread = 0; thread < threadsNum - 1; thread++ )
        phiOmegaFut.push_back( async(&PiecewiseLinearFunction::partialPhiOmega, this, thread, compByThread) );

    std::vector<Formula> phiOmegaFirst;
    for ( size_t i = (threadsNum - 1) * compByThread; i < linearPieceCollection.size(); i++ )
    {
        phiOmegaFirst.push_back( linearPieceCollection.at(i).getRepresentationModsat().phi );

        for ( size_t k = 0; k < linearPieceCollection.size(); k++ )
            if ( k != i )
                if ( linearPieceCollection.at(i).comparedIsAbove(linearPieceCollection.at(k)) )
                    phiOmegaFirst.back().addMinimum(linearPieceCollection.at(k).getRepresentationModsat().phi);
    }

    latticeFormula = phiOmegaFirst.at(0);
    for ( size_t i = 1; i < phiOmegaFirst.size(); i++ )
        latticeFormula.addMaximum(phiOmegaFirst.at(i));

    for ( unsigned thread = 0; thread < threadsNum - 1; thread++ )
    {
        std::vector<Formula> phiOmega = phiOmegaFut.at(thread).get();

        for ( size_t i = 0; i < phiOmega.size(); i++ )
            latticeFormula.addMaximum(phiOmega.at(i));
    }
}

void PiecewiseLinearFunction::representModsat()
{
    representPiecesModSat();

    if ( processingMode == Multi )
        representLatticeFormula(std::thread::hardware_concurrency());
    else if ( processingMode == Single )
        representLatticeFormula(1);

    modsatTranslation = true;
}

void PiecewiseLinearFunction::equivalentTo(Variable variable)
{
    if ( !modsatTranslation )
        representModsat();

    latticeFormula.addEquivalence( lukaFormula::Formula(variable) );
}

std::vector<RegionalLinearPiece> PiecewiseLinearFunction::getLinearPieceCollection()
{
    if ( !modsatTranslation )
        representModsat();

    return linearPieceCollection;
}

Formula PiecewiseLinearFunction::getLatticeFormula()
{
    if ( !modsatTranslation )
        representModsat();

    return latticeFormula;
}

void PiecewiseLinearFunction::printLimodsatFile()
{
    if ( !modsatTranslation )
        representModsat();

    std::ofstream outputFile(outputFileName);

    //outputFile << "-= Formula phi =-" << std::endl << std::endl;
    outputFile << "-= Formula phi =- MAXVAR " << var->currentVariable() << std::endl << std::endl;
    latticeFormula.print(&outputFile);

    outputFile << std::endl << "-= MODSAT Set Phi =-" << std::endl;

    for ( size_t i = 0; i < linearPieceCollection.size(); i++ )
    {
        outputFile << std::endl << "-= Linear Piece " << i+1 << " =-" << std::endl;
        linearPieceCollection.at(i).printModsatSet(&outputFile);
    }
}
}
