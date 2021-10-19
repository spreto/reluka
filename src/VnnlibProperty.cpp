#include <iostream>
#include <vector>
#include <cmath>
#include "VnnlibProperty.h"
#include "NeuralNetwork.h"
#include "LinearPiece.h"

namespace reluka
{
VnnlibProperty::VnnlibProperty(std::string vnnlibFileName,
                               pwl2limodsat::VariableManager *varMan) :
                               variableManager(varMan)
{
    vnnlibFile.open(vnnlibFileName);

    if ( !vnnlibFile.is_open() )
        throw std::invalid_argument("Unable to open vnnlib file.");

    if ( vnnlibFile.eof() )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    if ( vnnlibFileName.substr(vnnlibFileName.size()-7,7) == ".vnnlib" )
        propertyFileName = vnnlibFileName.substr(0,vnnlibFileName.size()-7);
    else
        propertyFileName = vnnlibFileName;

    propertyFileName.append(".liprop");

    getline(vnnlibFile, currentVnnlibLine);
    nextNonSpace();
}

VnnlibProperty::~VnnlibProperty()
{
    vnnlibFile.close();
}

bool VnnlibProperty::nextNonSpace()
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

void VnnlibProperty::parseVnnlibDeclareConst()
{
    bool inputVar;
    unsigned testingDim;

    if ( currentVnnlibLine.compare(currentLinePosition, 2, "X_") == 0 )
    {
        inputVar = true;
        testingDim = nnInputDimension;
    }
    else if ( currentVnnlibLine.compare(currentLinePosition, 2, "Y_") == 0 )
    {
        inputVar = false;
        testingDim = nnOutputDimension;
    }
    else
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 2;
    size_t blockLength = currentVnnlibLine.find_first_of(" ", currentLinePosition) - currentLinePosition;

    if ( currentVnnlibLine.compare(currentLinePosition,
                                   blockLength,
                                   std::to_string(testingDim)) == 0 )
    {
        if ( inputVar )
            nnInputDimension++;
        else
            nnOutputDimension++;
    }
    else
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += blockLength;
    nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 4, "Real") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 4;
    nextNonSpace();
}

std::pair<VnnlibProperty::AssertType,lukaFormula::Modsat> VnnlibProperty::parseVnnlibAtomicAssert(AtomicAssertType atomicAssertType)
{
    size_t beginPos[2], stringLength[2];
    std::string curLine[2];

    curLine[0] = currentVnnlibLine;
    beginPos[0] = currentLinePosition;
    stringLength[0] = std::fmin( currentVnnlibLine.find_first_of(" ", beginPos[0]),
                                 currentVnnlibLine.size() ) - beginPos[0];
    currentLinePosition = beginPos[0] + stringLength[0];
    nextNonSpace();

    curLine[1] = currentVnnlibLine;
    beginPos[1] = currentLinePosition;
    stringLength[1] = std::fmin( std::fmin( currentVnnlibLine.find_first_of(" ", beginPos[1]),
                                            currentVnnlibLine.find_first_of(")", beginPos[1]) ),
                                 currentVnnlibLine.size() ) - beginPos[1];
    currentLinePosition = beginPos[1] + stringLength[1];
    nextNonSpace();

    AssertType assertType;
    lukaFormula::Formula form[2];
    lukaFormula::ModsatSet msSet;

    for ( unsigned i = 0; i < 2; i++ )
    {
        if ( ( curLine[i].compare(beginPos[i], 2, "X_") == 0 ) ||
             ( curLine[i].compare(beginPos[i], 2, "Y_") == 0 ) )
        {
            unsigned varNum = stoi(curLine[i].substr(beginPos[i]+2, stringLength[i]-2));

            if ( curLine[i].compare(beginPos[i], 2, "X_") == 0 )
            {
                if ( varNum < nnInputDimension )
                {
                    assertType = Input;
                    form[i] = lukaFormula::Formula(pwl2limodsat::Variable(varNum+1));
                }
                else
                    throw std::invalid_argument("Not in standard vnnlib file format.");
            }
            else
            {
                if ( varNum < nnOutputDimension )
                {
                    assertType = Output;
                    pwl2limodsat::Variable auxVar;

                    std::map<unsigned,pwl2limodsat::Variable>::iterator it = nnOutputInfo.find(varNum);
                    if ( it == nnOutputInfo.end() )
                        auxVar = variableManager->newVariable();
                    else
                        auxVar = it->second;

                    form[i] = lukaFormula::Formula(auxVar);
                    nnOutputInfo[varNum] = auxVar;
                }
                else
                    throw std::invalid_argument("Not in standard vnnlib file format.");
            }
        }
        else
        {
            double constDouble = std::stod(curLine[i].substr(beginPos[i], stringLength[i]));
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

std::pair<VnnlibProperty::AssertType,lukaFormula::Modsat> VnnlibProperty::parseVnnlibAssert()
{
    std::pair<AssertType,lukaFormula::Modsat> assertPair;
    assertPair.first = Undefined;

    if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 1;
    nextNonSpace();

    if ( currentVnnlibLine.compare(currentLinePosition, 3, "and") == 0 )
    {
        currentLinePosition += 3;
        nextNonSpace();

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
        nextNonSpace();

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
        throw std::invalid_argument("Not in standard vnnlib file format.");

    if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
        throw std::invalid_argument("Not in standard vnnlib file format.");

    currentLinePosition += 1;
    nextNonSpace();

    return assertPair;
}

void VnnlibProperty::readDeclarations()
{
    bool finished = false;

    while ( !finished )
    {
        if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
            throw std::invalid_argument("Not in standard vnnlib file format.");
        else
        {
            currentLinePosition++;
            nextNonSpace();
        }

        if ( currentVnnlibLine.compare(currentLinePosition, 13, "declare-const") == 0 )
        {
            currentLinePosition += 13;

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

void VnnlibProperty::vnnlib2property()
{
    bool firstIteration = true;
    bool finished = false;

    while ( !finished )
    {
        if ( firstIteration )
            firstIteration = false;
        else
        {
            if ( currentVnnlibLine.compare(currentLinePosition, 1, "(") != 0 )
                finished = true;
            else
            {
                currentLinePosition++;
                nextNonSpace();
            }
        }

        if ( !finished && currentVnnlibLine.compare(currentLinePosition, 6, "assert") == 0 )
        {
            currentLinePosition += 6;

            if ( !nextNonSpace() )
                throw std::invalid_argument("Not in standard vnnlib file format.");
            else
            {
                lukaFormula::Modsat assertReturn = parseVnnlibAssert().second;

                if ( currentVnnlibLine.compare(currentLinePosition, 1, ")") != 0 )
                    throw std::invalid_argument("Not in standard vnnlib file format.");
                else
                {
                    currentLinePosition++;
                    nextNonSpace();
                }

                propertyFormulas.push_back(assertReturn.phi);

                propertyFormulas.insert(propertyFormulas.end(),
                                        assertReturn.Phi.begin(),
                                        assertReturn.Phi.end());
            }
        }
    }
}

void VnnlibProperty::buildVnnlibProperty()
{
    if ( !propertyBuilding )
    {
        readDeclarations();
        variableManager->setDimension(nnInputDimension);
        vnnlib2property();
        propertyBuilding = true;
    }
}

void VnnlibProperty::setOutputAddresses(std::vector<pwl2limodsat::PiecewiseLinearFunction> *pwlAddresses)
{
    if ( !propertyBuilding )
        buildVnnlibProperty();

    if ( pwlAddresses->size() != nnOutputInfo.size() )
        throw std::invalid_argument("Different number of pwl functions than of outputs declared in the vnnlib file.");

    for ( size_t i = 0; i < nnOutputInfo.size(); i++ )
        pwlAddresses->at(i).equivalentTo(nnOutputInfo.at(i));

    nnOutputAddresses = pwlAddresses;
}

void VnnlibProperty::buildNnOutputIndexes()
{
    if ( !propertyBuilding )
        buildVnnlibProperty();

    if ( nnOutputIndexes.empty() )
    {
        for ( unsigned i = 0; i < nnOutputDimension; i++ )
            if ( nnOutputInfo.count(i) != 0 )
                nnOutputIndexes.push_back(i);
    }
}

std::vector<unsigned> VnnlibProperty::getNnOutputIndexes()
{
    if ( nnOutputIndexes.empty() )
        buildNnOutputIndexes();

    return nnOutputIndexes;
}

pwl2limodsat::Variable VnnlibProperty::getVariable(unsigned nnOutputIdx)
{
    if ( !propertyBuilding )
        buildVnnlibProperty();

    if ( nnOutputInfo.count(nnOutputIdx) == 0 )
        throw std::invalid_argument("Vnnlib file does not refer to such output index.");

    return nnOutputInfo[nnOutputIdx];
}

void VnnlibProperty::printLipropFile()
{
    if ( !propertyBuilding )
        buildVnnlibProperty();

    if ( nnOutputDimension != nnOutputAddresses->size() )
        throw std::invalid_argument("Pwl addresses are not coherent.");

    std::ofstream propertyFile(propertyFileName);
    propertyFile << "Sat" << std::endl << std::endl;

    for ( pwl2limodsat::PiecewiseLinearFunction pwl : *nnOutputAddresses )
    {
        for ( pwl2limodsat::RegionalLinearPiece rlp : pwl.getLinearPieceCollection() )
            rlp.printModsatSetAs(&propertyFile, "f:");

        propertyFile << "f:" << std::endl;
        pwl.getLatticeFormula().print(&propertyFile);
    }

    for ( lukaFormula::Formula pForm : propertyFormulas )
    {
        propertyFile << "f:" << std::endl;
        pForm.print(&propertyFile);
    }
}
}
