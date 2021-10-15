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
        throw std::invalid_argument("Not in standard vnnlib file format.1");

    if ( vnnlibFileName.substr(vnnlibFileName.size()-7,7) == ".vnnlib" )
        propertyFileName = vnnlibFileName.substr(0,vnnlibFileName.size()-7);
    else
        propertyFileName = vnnlibFileName;

    propertyFileName.append(".liprop");

    getline(vnnlibFile, currentVnnlibLine);
    nextNonSpace();
}

Property::~Property()
{
    vnnlibFile.close();
}

bool Property::nextNonSpace()
{
    bool finished = false;
    bool found;

    while ( !finished )
    {
        if ( currentVnnlibLine.compare(0, 1, ";") == 0 ||
             currentVnnlibLine.empty() ||
             currentLinePosition == currentVnnlibLine.size() ||
             currentLinePosition == std::string::npos )
        {
            if ( vnnlibFile.eof() )
            {
                finished = true;
                found = false;
            }
            else
            {
                currentLinePosition = 0;
                getline(vnnlibFile, currentVnnlibLine);
            }
        }
        else
        {
            currentLinePosition = currentVnnlibLine.find_first_not_of(" ", currentLinePosition);
            if ( currentLinePosition != std::string::npos )
            {
                finished = true;
                found = true;
            }
        }
    }

    return found;
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
        throw std::invalid_argument("Not in standard vnnlib file format.1,5");

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
        throw std::invalid_argument("Not in standard vnnlib file format.2");

    currentLinePosition += blockLength;
    nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 4, "Real") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.3");

    currentLinePosition += 4;
    nextNonSpace();
}

std::pair<Property::AssertType,lukaFormula::Modsat> Property::parseVnnlibAtomicAssert(AtomicAssertType atomicAssertType)
{
    size_t beginPos[2], stringLength[2];

    beginPos[0] = currentLinePosition;
    stringLength[0] = currentVnnlibLine.find_first_of(" ", beginPos[0]) - beginPos[0];
    currentLinePosition = beginPos[0] + stringLength[0];
    nextNonSpace();
    beginPos[1] = currentLinePosition;
    stringLength[1] = std::fmin( currentVnnlibLine.find_first_of(" ", beginPos[1]),
                                 currentVnnlibLine.find_first_of(")", beginPos[1]) ) - beginPos[1];
    currentLinePosition = beginPos[1] + stringLength[1];
    nextNonSpace();

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
                    form[i] = lukaFormula::Formula(pwl2limodsat::Variable(varNum+1));
                }
                else
                    throw std::invalid_argument("Not in standard vnnlib file format.4");
            }
            else
            {
                if ( varNum < outputDimension )
                {
                    assertType = Output;
                    pwl2limodsat::Variable auxVar;

                    std::map<unsigned,pwl2limodsat::Variable>::iterator it = outputInfo.find(varNum);
                    if ( it == outputInfo.end() )
                        auxVar = variableManager->newVariable();
                    else
                        auxVar = it->second;

                    form[i] = lukaFormula::Formula(auxVar);
                    outputInfo[varNum] = auxVar;
                }
                else
                    throw std::invalid_argument("Not in standard vnnlib file format.5");
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
        throw std::invalid_argument("Not in standard vnnlib file format.6");

    currentLinePosition += 1;
    nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 3, "and") == 0 )
    {
        currentLinePosition += 3;
        nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.7");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        {
            std::pair<AssertType,lukaFormula::Modsat> returnPair = parseVnnlibAssert();

            if ( assertPair.first == Undefined )
                assertPair.first = returnPair.first;
            else if ( assertPair.first != returnPair.first )
                throw std::invalid_argument("Not in standard vnnlib file format.8");

            assertPair.second.phi.addMinimum( returnPair.second.phi );
            assertPair.second.Phi.insert(assertPair.second.Phi.end(),
                                         returnPair.second.Phi.begin(),
                                         returnPair.second.Phi.end());
        }
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "or") == 0 )
    {
        currentLinePosition += 2;
        nextNonSpace();

        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.9");

        while ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        {
            std::pair<AssertType,lukaFormula::Modsat> returnPair = parseVnnlibAssert();

            if ( assertPair.first == Undefined )
                assertPair.first = returnPair.first;
            else if ( assertPair.first != returnPair.first )
                throw std::invalid_argument("Not in standard vnnlib file format.10");

            assertPair.second.phi.addMaximum( returnPair.second.phi );
            assertPair.second.Phi.insert(assertPair.second.Phi.end(),
                                         returnPair.second.Phi.begin(),
                                         returnPair.second.Phi.end());
        }
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "<=") == 0 )
    {
        currentLinePosition += 2;
        nextNonSpace();
        assertPair = parseVnnlibAtomicAssert(LessEq);
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, ">=") == 0 )
    {
        currentLinePosition += 2;
        nextNonSpace();
        assertPair = parseVnnlibAtomicAssert(GreaterEq);
    }
    else
        throw std::invalid_argument("Not in standard vnnlib file format.11");

    if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.12");

    currentLinePosition += 1;
    nextNonSpace();

    return assertPair;
}

void Property::readDeclarations()
{
    bool finished = false;

    while ( !finished )
    {
        if ( currentVnnlibLine.compare(currentLinePosition, 14, "(declare-const") == 0 )
        {
            currentLinePosition += 14;

            if ( !nextNonSpace() )
                throw std::invalid_argument("Not in standard vnnlib file format.");
            else
            {
                parseVnnlibDeclareConst();
                if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
                    throw std::invalid_argument("Not in standard vnnlib file format.");
                else
                {
                    currentLinePosition++;
                    if ( !nextNonSpace() )
                        throw std::invalid_argument("No property declared.");
                }
            }
        }
        else
            finished = true;
    }
}

void Property::vnnlib2property()
{
    bool finished = false;

    while ( !finished )
    {
        if ( currentVnnlibLine.compare(currentLinePosition, 7, "(assert") == 0 )
        {
            currentLinePosition += 7;

            if ( !nextNonSpace() )
                throw std::invalid_argument("Not in standard vnnlib file format.");
            else
            {
                std::pair<AssertType,lukaFormula::Modsat> assertReturn = parseVnnlibAssert();

                if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
                    throw std::invalid_argument("Not in standard vnnlib file format.");
                else
                {
                    currentLinePosition++;
                    nextNonSpace();
                }

                if ( assertReturn.first == Input )
                    premiseFormulas.push_back(assertReturn.second.phi);
                else
                    conclusionFormula = assertReturn.second.phi;

                premiseFormulas.insert(premiseFormulas.end(),
                                       assertReturn.second.Phi.begin(),
                                       assertReturn.second.Phi.end());
            }
        }
        else
            finished = true;
    }
}

void Property::buildProperty()
{
    if ( !propertyBuilding )
    {
        readDeclarations();
        variableManager->setDimension(inputDimension);
        vnnlib2property();
        propertyBuilding = true;
    }
}

void Property::setOutputAddress(pwl2limodsat::PiecewiseLinearFunction* pwlAddress)
{
    outputAddresses.push_back(pwlAddress);
}

void Property::printLipropFile()
{
    if ( !propertyBuilding )
        buildProperty();

    if ( outputDimension != outputAddresses.size() )
        throw std::invalid_argument("Pwl addresses not found.");

    std::ofstream propertyFile(propertyFileName);
    propertyFile << "Cons" << std::endl << std::endl;

    for ( pwl2limodsat::PiecewiseLinearFunction* pwl : outputAddresses )
    {
        for ( pwl2limodsat::RegionalLinearPiece rlp : pwl->getLinearPieceCollection() )
            rlp.printModsatSetAs(&propertyFile, "P1:");

        propertyFile << "P2:" << std::endl;
        pwl->getLatticeFormula().print(&propertyFile);
    }

    for ( lukaFormula::Formula pForm : premiseFormulas )
    {
        propertyFile << "P3:" << std::endl;
        pForm.print(&propertyFile);
    }

    propertyFile << std::endl << "C:" << std::endl;
    conclusionFormula.print(&propertyFile);
}
}
