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

#include "InputParser.h"

namespace pwl2limodsat
{
InputParser::InputParser(const char* inputPwlFileName) :
    pwlFileName(inputPwlFileName)
{
    pwlFile.open(pwlFileName);

    if ( !pwlFile.is_open() )
        throw std::invalid_argument("Unable to open pwl file.");
    else
    {
        nextLine();
        if ( currentLine.compare(0,3,"tl ") == 0)
            mode = TL;
        else if ( currentLine.compare(0,3,"pwl") == 0 )
            mode = PWL;
        else
            throw std::invalid_argument("Not in standard pwl file format.");
    }
}

InputParser::~InputParser()
{
    pwlFile.close();
}

void InputParser::nextLine()
{
    getline(pwlFile,currentLine);

    while ( ( ( currentLine.compare(0,1,"c") == 0 ) || ( currentLine.empty() ) ) && !pwlFile.eof() )
        getline(pwlFile, currentLine);
}

LinearPieceData InputParser::readLinearPiece(unsigned beginingPosition)
{
    LinearPieceData lpData;
    size_t beginPosition = 0, endPosition;
    LPCoefInteger numerator;
    LPCoefNonNegative denumerator;

    do
    {
        beginPosition = ( beginPosition == 0 ? beginingPosition : endPosition+1 );
        endPosition = currentLine.find_first_of(" ", beginPosition);

        if ( (beginPosition > currentLine.size()) || (endPosition > currentLine.size()) )
            throw std::invalid_argument("Not in standard pwl file format.");
        else
        {
            numerator = stoi( currentLine.substr( beginPosition, endPosition-beginPosition ) );
            beginPosition = endPosition+1;
            endPosition = currentLine.find_first_of(" ", beginPosition);
        }
        if ( beginPosition > currentLine.size() )
            throw std::invalid_argument("Not in standard pwl file format.");
        else
        {
            denumerator = stoi( currentLine.substr( beginPosition, endPosition-beginPosition ) );
            if ( denumerator < 1 )
                throw std::out_of_range("Fraction denumerator must be positive.");
            else
                lpData.push_back( LinearPieceCoefficient( numerator, (unsigned) denumerator ) );
        }
    } while ( endPosition < currentLine.size() );

    return lpData;
}

void InputParser::buildTlInstance()
{
    tlData = readLinearPiece(3);
}

void InputParser::buildPwlInstance()
{
    BoundaryPrototype boundProt;
    LinearPieceData lpData;
    BoundaryCollection boundaryCollection;

    size_t beginPosition = 0, endPosition;
    unsigned boundaryCounter = 0;
    unsigned boundaryNumber;

    nextLine();

    while ( !pwlFile.eof() )
    {
        if ( currentLine.compare(0,2,"b ") == 0 )
        {
            do
            {
                beginPosition = ( beginPosition == 0 ? 2 : endPosition+1 );
                endPosition = currentLine.find_first_of(" ", beginPosition);

                if ( beginPosition > currentLine.size() )
                    throw std::invalid_argument("Not in standard pwl file format.");
                else
                    boundProt.push_back(stof(currentLine.substr(beginPosition, endPosition-beginPosition)));
            } while ( endPosition < currentLine.size() );

            boundaryPrototypeData.push_back(boundProt);
            boundProt.clear();
            boundaryCounter++;
            beginPosition = 0;

            nextLine();
        }
        else if ( currentLine.compare(0,2,"p ") == 0 )
        {
            lpData = readLinearPiece(2);
            nextLine();

            while ( (currentLine.compare(0,2,"g ") == 0) || (currentLine.compare(0,2,"l ") == 0) )
            {
                if ( currentLine.size() < 3 )
                    throw std::invalid_argument("Not in standard pwl file format.");
                else
                    boundaryNumber = (unsigned) stoul(currentLine.substr(2, currentLine.size()-2));

                if ( (boundaryNumber > 0) && (boundaryNumber <= boundaryCounter) )
                {
                    if ( currentLine.compare(0,2,"g ") == 0 )
                        boundaryCollection.push_back(Boundary(boundaryNumber-1, GeqZero));
                    else if ( currentLine.compare(0,2,"l ") == 0 )
                        boundaryCollection.push_back(Boundary(boundaryNumber-1, LeqZero));
                }
                else
                    throw std::out_of_range("Nonexistent boundary prototype.");

                nextLine();
            }

            RegionalLinearPieceData rlpData = { lpData, boundaryCollection };
            pwlData.push_back(rlpData);
            lpData.clear();
            boundaryCollection.clear();
        }
        else
            throw std::invalid_argument("Not in standard pwl file format.");
    }

    unsigned dim = boundaryPrototypeData.at(0).size();

    for ( size_t i = 1; i < boundaryPrototypeData.size(); i++ )
        if ( boundaryPrototypeData.at(i).size() != dim )
            throw std::invalid_argument("Dimension inconsistency.");

    for ( size_t i = 0; i < pwlData.size(); i++ )
        if ( pwlData.at(i).lpData.size() != dim )
            throw std::invalid_argument("Dimension inconsistency.");

    pwlTranslation = true;
}

LinearPieceData InputParser::getTlInstanceData()
{
    if ( mode != TL )
        throw std::domain_error("Not a truncated linear function.");

    if ( !tlTranslation )
        buildTlInstance();

    return tlData;
}

PiecewiseLinearFunctionData InputParser::getPwlInstanceData()
{
    if ( mode != PWL )
        throw std::domain_error("Not a general piecewise linear function.");

    if ( !pwlTranslation )
        buildPwlInstance();

    return pwlData;
}

BoundaryPrototypeCollection InputParser::getPwlInstanceBoundProt()
{
    if ( mode != PWL )
        throw std::domain_error("Not a general piecewise linear function.");

    if ( !pwlTranslation )
        buildPwlInstance();

    return boundaryPrototypeData;
}
}
