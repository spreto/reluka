#include <iostream>
#include <vector>
#include "Property.h"

namespace reluka
{
Property::Property(const char* inputVnnlibFileName) :
    vnnlibFileName(inputVnnlibFileName)
{
    vnnlibFile.open(vnnlibFileName);

    if ( !vnnlibFile.is_open() )
        throw std::invalid_argument("Unable to open vnnlib file.");

    if ( vnnlibFile.eof() )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    while ( !vnnlibFile.eof() &&
             ( (currentVnnlibLine.compare(0, 1, ";") == 0) ||
               (currentVnnlibLine.empty()) ) )
        getline(vnnlibFile, currentVnnlibLine);
}

Property::~Property()
{
    vnnlibFile.close();
}

size_t Property::nextNonSpace()
{
    if ( currentLinePosition == currentVnnlibLine.size() )
    {
        currentLinePosition = 0;

        if ( !vnnlibFile.eof() )
            getline(vnnlibFile, currentVnnlibLine);
    }

    size_t nonSpace = currentVnnlibLine.find_first_not_of(" ", currentLinePosition);

    while ( nonSpace == std::string::npos )
    {
        if ( !vnnlibFile.eof() )
        {
            while ( !vnnlibFile.eof() &&
                    ( (currentVnnlibLine.compare(0, 1, ";") == 0) ||
                      (currentVnnlibLine.empty()) ) )
                getline(vnnlibFile, currentVnnlibLine);

            nonSpace = currentVnnlibLine.find_first_not_of(" ", 0);
        }
        else
            return std::string::npos;
    }

    return nonSpace;
}

void Property::parseVnnlibDeclareConst()
{
    bool inputVar;
    unsigned testingDim;

    if ( currentVnnlibLine.compare(currentLinePosition, 2, "X_") == 0 )
    {
        inputVar = true;
        testingDim = inputDimension;
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "Y_") == 0 )
    {
        inputVar = false;
        testingDim = outputDimension;
    }
    else
        throw std::invalid_argument("Not in standard vnnlib file format");

    currentLinePosition += 2;
    size_t blockLength = currentVnnlibLine.find_first_of(" ", currentLinePosition) - currentLinePosition;

    if ( currentVnnlibLine.compare(currentLinePosition,
                                   blockLength,
                                   std::to_string(testingDim)) == 0 )
    {
        if ( inputVar )
            inputDimension++;
        else
            outputDimension++;
    }
    else { std::cout << testingDim << "   " << inputDimension << std::endl;
        throw std::invalid_argument("Not in standard vnnlib file format."); }

    currentLinePosition += blockLength;
    currentLinePosition = nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 4, "Real") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 4;
    currentLinePosition = nextNonSpace();
}

lukaFormula::Formula Property::parseVnnlibInequality(AtomicFormulaType type)
{
    lukaFormula::Formula assertAtomicForm;

    // lembra dos espaços em branco... só pra funcionar por hj
    currentLinePosition = currentVnnlibLine.find_first_of(")");
    //////

    return assertAtomicForm;
}

lukaFormula::Formula Property::parseVnnlibAssert()
{
    lukaFormula::Formula assertForm;

    if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.4");

    currentLinePosition += 1;
    currentLinePosition = nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 3, "and") == 0 )
    {
        currentLinePosition += 3;
        currentLinePosition = nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.5");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
            assertForm.addMinimum( parseVnnlibAssert() );
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "or") == 0 )
    {
        currentLinePosition += 3;
        currentLinePosition = nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.6");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
            assertForm.addMaximum( parseVnnlibAssert() );
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "<=") == 0 )
        assertForm = parseVnnlibInequality(LessEq);
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, ">=") == 0 )
        assertForm = parseVnnlibInequality(GreaterEq);
    else
        throw std::invalid_argument("Not in standard vnnlib file format.7");

    if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.8");

    currentLinePosition += 1;
    currentLinePosition = nextNonSpace();

    return assertForm;
}

void Property::vnnlib2property()
{
    currentLinePosition = 0;
    currentLinePosition = nextNonSpace();

    while ( currentLinePosition != std::string::npos )
    {
        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.1");

        currentLinePosition += 1;
        currentLinePosition = nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 13, "declare-const") == 0 )
        {
            currentLinePosition += 13;
            currentLinePosition = nextNonSpace();
            parseVnnlibDeclareConst();
        }
        else if ( currentVnnlibLine.compare(currentLinePosition, 6, "assert") == 0 )
        {
            currentLinePosition += 6;
            currentLinePosition = nextNonSpace();
            assertFormulas.push_back(parseVnnlibAssert());
        }
        else
            throw std::invalid_argument("Not in standard vnnlib file format.2");

        if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.3");

        currentLinePosition += 1;
        currentLinePosition = nextNonSpace();
    }
}

void Property::buildProperty()
{
    vnnlib2property();
}
}
