#include <iostream>
#include <vector>
#include <cmath>
#include "Property.h"
#include "NeuralNetwork.h"
#include "LinearPiece.h"

namespace reluka
{
Property::Property(const char* inputVnnlibFileName,
                   pwl2limodsat::VariableManager *varMan) :
    vnnlibFileName(inputVnnlibFileName),
    variableManager(varMan)
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
    else
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += blockLength;
    currentLinePosition = nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 4, "Real") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 4;
    currentLinePosition = nextNonSpace();
}

std::pair<Property::AssertType,lukaFormula::Modsat> Property::parseVnnlibAtomicAssert(AtomicAssertType atomicAssertType)
{
    size_t beginPos[2], stringLength[2];

    beginPos[0] = currentLinePosition;
    stringLength[0] = currentVnnlibLine.find_first_of(" ", beginPos[0]) - beginPos[0];
    currentLinePosition = beginPos[0] + stringLength[0];
    currentLinePosition = nextNonSpace();
    beginPos[1] = currentLinePosition;
    stringLength[1] = std::fmin( currentVnnlibLine.find_first_of(" ", beginPos[1]),
                                 currentVnnlibLine.find_first_of(")", beginPos[1]) ) - beginPos[1];
    currentLinePosition = beginPos[1] + stringLength[1];
    currentLinePosition = nextNonSpace();

    AssertType assertType;
    lukaFormula::Formula form[2];
    lukaFormula::ModsatSet msSet;

    for ( unsigned i = 0; i < 2; i++ )
    {
        if ( ( currentVnnlibLine.compare(beginPos[i], 2, "X_") == 0 ) ||
             ( currentVnnlibLine.compare(beginPos[i], 2, "Y_") == 0 ) )
        {
            unsigned varNum = stoi(currentVnnlibLine.substr(beginPos[i]+2, stringLength[i]-2));

            if ( currentVnnlibLine.compare(beginPos[i], 2, "X_") == 0 )
            {
                if ( varNum < inputDimension )
                {
                    assertType = Input;
                    form[i] = lukaFormula::Formula(varNum+1);
                }
                else
                    std::invalid_argument("Not in standard vnnlib file format.");
            }
            else
            {
                if ( varNum < outputDimension )
                {
                    assertType = Output;
                    form[i] = lukaFormula::Formula(0); // provisório
                }
                else
                    std::invalid_argument("Not in standard vnnlib file format.");
            }
        }
        else
        {
            double constDouble = std::stod(currentVnnlibLine.substr(beginPos[i], stringLength[i]));
            pwl2limodsat::LinearPieceCoefficient constFraction = reluka::NeuralNetwork::dec2frac(constDouble);

            if ( !variableManager->isThereConstant(constFraction.second) )
                msSet = pwl2limodsat::LinearPiece::defineConstant(variableManager, constFraction.second);

            lukaFormula::Modsat msAux = pwl2limodsat::LinearPiece::multiplyConstant(variableManager,
                                                                                    constFraction.first,
                                                                                    constFraction.second);
            form[i] = msAux.phi;
            msSet.insert(msSet.end(), msAux.Phi.begin(), msAux.Phi.end());
        }
    }

    if ( atomicAssertType == LessEq )
    {
        form[0].addImplication(form[1]);
        return std::pair<AssertType,lukaFormula::Modsat>(assertType, { form[0], msSet });
    }
    else
    {
        form[1].addImplication(form[0]);
        return std::pair<AssertType,lukaFormula::Modsat>(assertType, { form[1], msSet });
    }
}

std::pair<Property::AssertType,lukaFormula::Modsat> Property::parseVnnlibAssert()
{
    std::pair<AssertType,lukaFormula::Modsat> assertPair;
    assertPair.first = Undefined;

    if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 1;
    currentLinePosition = nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 3, "and") == 0 )
    {
        currentLinePosition += 3;
        currentLinePosition = nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        {
            std::pair<AssertType,lukaFormula::Modsat> returnPair = parseVnnlibAssert();

            if ( assertPair.first == Undefined )
                assertPair.first = returnPair.first;
            else if ( assertPair.first != returnPair.first )
                throw std::invalid_argument("Not in standard vnnlib file format.");

            assertPair.second.phi.addMinimum( returnPair.second.phi );
            assertPair.second.Phi.insert(assertPair.second.Phi.end(),
                                         returnPair.second.Phi.begin(),
                                         returnPair.second.Phi.end());
        }
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "or") == 0 )
    {
        currentLinePosition += 2;
        currentLinePosition = nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        {
            std::pair<AssertType,lukaFormula::Modsat> returnPair = parseVnnlibAssert();

            if ( assertPair.first == Undefined )
                assertPair.first = returnPair.first;
            else if ( assertPair.first != returnPair.first )
                throw std::invalid_argument("Not in standard vnnlib file format.");

            assertPair.second.phi.addMaximum( returnPair.second.phi );
            assertPair.second.Phi.insert(assertPair.second.Phi.end(),
                                         returnPair.second.Phi.begin(),
                                         returnPair.second.Phi.end());
        }
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "<=") == 0 )
    {
        currentLinePosition += 2;
        currentLinePosition = nextNonSpace();
        assertPair = parseVnnlibAtomicAssert(LessEq);
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, ">=") == 0 )
    {
        currentLinePosition += 2;
        currentLinePosition = nextNonSpace();
        assertPair = parseVnnlibAtomicAssert(GreaterEq);
    }
    else
        throw std::invalid_argument("Not in standard vnnlib file format.");

    if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 1;
    currentLinePosition = nextNonSpace();

    return assertPair;
}

void Property::vnnlib2property()
{
    currentLinePosition = 0;
    currentLinePosition = nextNonSpace();

    while ( currentLinePosition != std::string::npos )
    {
        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.");

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

            std::pair<AssertType,lukaFormula::Modsat> assertReturn = parseVnnlibAssert();

            if ( assertReturn.first == Input )
                premiseFormulas.push_back(assertReturn.second.phi);
            else
                conclusionFormula = assertReturn.second.phi;

            premiseFormulas.insert(premiseFormulas.end(),
                                   assertReturn.second.Phi.begin(),
                                   assertReturn.second.Phi.end());
        }
        else
            throw std::invalid_argument("Not in standard vnnlib file format.");

        if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.");

        currentLinePosition += 1;
        currentLinePosition = nextNonSpace();
    } std::cout << inputDimension << " | " << outputDimension << std::endl;
}

void Property::buildProperty()
{
    if ( !propertyBuilding )
        vnnlib2property();
}

void Property::print()
{
    std::ofstream file("test.out");

    for ( lukaFormula::Formula prem : premiseFormulas )
    {
        file << "prem:" << std::endl;
        prem.print(&file);
        file << std::endl;
    }

    file << "conclus:" << std::endl;
    conclusionFormula.print(&file);
}
}
