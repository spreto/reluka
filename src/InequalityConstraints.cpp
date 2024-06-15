#include "InequalityConstraints.h"
#include "LinearPiece.h"

#include <iostream>

namespace reluka
{
InequalityConstraints::InequalityConstraints(std::string ineqconsFileName,
                                             pwl2limodsat::VariableManager *varMan) :
                                             variableManager(varMan)
{
    ineqconsFile.open(ineqconsFileName);

    if ( !ineqconsFile.is_open() )
        throw std::invalid_argument("Unable to open inequality constraints file.");

    if ( ineqconsFile.eof() )
        throw std::invalid_argument("Not in standard inequality constraints file format.");

    if ( ineqconsFileName.substr(ineqconsFileName.size()-9,9) == ".ineqcons" )
        propertyFileName = ineqconsFileName.substr(0,ineqconsFileName.size()-9);
    else
        propertyFileName = ineqconsFileName;

    propertyFileName.append(".liprop");
}

InequalityConstraints::~InequalityConstraints()
{
    ineqconsFile.close();
}

void InequalityConstraints::parseIneqcons()
{
    while ( !ineqconsFile.eof() )
    {
        getline(ineqconsFile, currentIneqconsLine);
        currentLinePosition = 0;

        if ( currentIneqconsLine.compare(currentLinePosition, 1, "x") == 0 )
        {
            currentLinePosition++;
            size_t blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned inputNum = stoi(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            double inputMin = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.size() - currentLinePosition;
            double inputMax = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            nnInputLimits[inputNum] = std::pair<double,double>(inputMin,inputMax);
        }
        else if ( currentIneqconsLine.compare(currentLinePosition, 2, "y>") == 0 )
        {
            currentLinePosition += 2;
            size_t blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned outputNum = stoi(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.size() - currentLinePosition;
            double center = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            nnOutputLimits[outputNum] = std::tuple<outputLimitsType,double,double>(GreaterEq,center,0);
            nnOutputIndexes.push_back(outputNum);
        }
        else if ( currentIneqconsLine.compare(currentLinePosition, 2, "y<") == 0 )
        {
            currentLinePosition += 2;
            size_t blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned outputNum = stoi(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.size() - currentLinePosition;
            double center = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            nnOutputLimits[outputNum] = std::tuple<outputLimitsType,double,double>(LessEq,center,0);
            nnOutputIndexes.push_back(outputNum);
        }
        else if ( currentIneqconsLine.compare(currentLinePosition, 1, "y") == 0 )
        {
            currentLinePosition++;
            size_t blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned outputNum = stoi(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            double outputMin = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqconsLine.size() - currentLinePosition;
            double outputMax = stod(currentIneqconsLine.substr(currentLinePosition, blockLenght));
            nnOutputLimits[outputNum] = std::tuple<outputLimitsType,double,double>(Both,outputMin,outputMax);
            nnOutputIndexes.push_back(outputNum);
        }
    }
}

void InequalityConstraints::buildIneqconsProperty()
{
    parseIneqcons();
    variableManager->setDimension(nnInputLimits.size());

    for ( auto& x : nnOutputIndexes )
        nnOutputVariables[x] = variableManager->newVariable();

    for ( auto& x : nnOutputLimits )
    {
        if ( std::get<0>(x.second) == GreaterEq )
        {
            if ( !variableManager->isThereConstant(2) )
                premises.push_back(pwl2limodsat::LinearPiece::defineConstant(variableManager, 2).back());
            lukaFormula::Formula auxForm(variableManager->constant(2));
            auxForm.addImplication(nnOutputVariables[x.first]);
            if ( conclusion.isEmpty() )
                conclusion = auxForm;
            else
                conclusion.addMinimum(auxForm);
        }
        else if ( std::get<0>(x.second) == LessEq )
        {}
        else if ( std::get<0>(x.second) == Both )
        {}
    }
}

void InequalityConstraints::setOutputAddresses(std::vector<pwl2limodsat::PiecewiseLinearFunction> *pwlAddresses)
{
    std::map<unsigned,pwl2limodsat::Variable>::iterator it = nnOutputVariables.begin();

    for ( size_t i = 0; i < nnOutputVariables.size(); i++ )
    {
        pwlAddresses->at(i).equivalentTo(it->second);
        it++;
    }

    nnOutputAddresses = pwlAddresses;
}
/* ou esse (testar qual funciona):
{
    for ( size_t i = 0; i < nnOutputVariables.size(); i++ )
        pwlAddresses->at(i).equivalentTo(nnOutputVariables.at(i));

    nnOutputAddresses = pwlAddresses;
}
*/
}
