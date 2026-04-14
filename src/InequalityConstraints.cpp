#include "InequalityConstraints.h"
#include "LinearPiece.h"
#include "NeuralNetwork.h"

#include <iostream>

namespace reluka
{
InequalityConstraints::InequalityConstraints(std::string ineqconsFileName,
                                             size_t inputDim,
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

    variableManager->setDimension(inputDim);
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
            nnOutputLimits[outputNum] = std::pair<double,double>(outputMin,outputMax);
            nnOutputIndexes.push_back(outputNum);
        }
    }

    ineqconsParsing = true;
}

void InequalityConstraints::buildIneqconsProperty( std::map<unsigned,std::pair<double,double>> originalOutputLim )
{
    if ( !ineqconsParsing )
        parseIneqcons();

    for ( auto& x : nnOutputIndexes )
        nnOutputVariables.push_back(variableManager->newVariable());

    for ( auto& lim : nnOutputLimits )
    {
        lim.second = std::pair<double,double>( (lim.second.first - originalOutputLim[lim.first].first) /
                                               (originalOutputLim[lim.first].second - originalOutputLim[lim.first].first),
                                               (lim.second.second - originalOutputLim[lim.first].first) /
                                               (originalOutputLim[lim.first].second - originalOutputLim[lim.first].first));

        pwl2limodsat::LinearPieceCoefficient leftLim = reluka::NeuralNetwork::dec2frac(lim.second.first);
        pwl2limodsat::LinearPieceCoefficient rightLim = reluka::NeuralNetwork::dec2frac(lim.second.second);

        if ( !variableManager->isThereConstant(leftLim.second) )
            premises.push_back(pwl2limodsat::LinearPiece::defineConstant(variableManager, leftLim.second).back());
        lukaFormula::Modsat auxModsat = pwl2limodsat::LinearPiece::multiplyConstant(variableManager, leftLim.first, leftLim.second);
        premises.insert(premises.end(), auxModsat.Phi.begin(), auxModsat.Phi.end());
        auxModsat.phi.addImplication(lukaFormula::Formula(nnOutputVariables[lim.first]));
        if ( conclusion.isEmpty() )
            conclusion = auxModsat.phi;
        else
            conclusion.addMinimum(auxModsat.phi);

        if ( !variableManager->isThereConstant(rightLim.second) )
            premises.push_back(pwl2limodsat::LinearPiece::defineConstant(variableManager, rightLim.second).back());
        auxModsat = pwl2limodsat::LinearPiece::multiplyConstant(variableManager, rightLim.first, rightLim.second);
        premises.insert(premises.end(), auxModsat.Phi.begin(), auxModsat.Phi.end());
        lukaFormula::Formula auxForm(nnOutputVariables[lim.first]);
        auxForm.addImplication(auxModsat.phi);
        conclusion.addMinimum(auxForm);
    }
/*
    for ( minimais )

    for ( maximais )
*/

    ineqconsProperty = true;
}

std::map<unsigned,std::pair<double,double>> InequalityConstraints::getInputLimits()
{
    if ( !ineqconsParsing )
        parseIneqcons();

    return nnInputLimits;
}

std::vector<unsigned> InequalityConstraints::getNnOutputIndexes()
{
    if ( !ineqconsParsing )
        parseIneqcons();

    return nnOutputIndexes;
}

void InequalityConstraints::printLiproperty(NeuralNetworkModSat *nnms)
{
    if ( !ineqconsProperty )
        buildIneqconsProperty(nnms->getOriginalOutputLim());

    std::ofstream propertyFile(propertyFileName);
    propertyFile << "Cons" << std::endl << std::endl;

    nnms->printNNmodsat(&propertyFile, nnOutputVariables);

    for ( lukaFormula::Formula form : premises )
    {
        propertyFile << "f:" << std::endl;
        form.print(&propertyFile);
    }

    propertyFile << "C:" << std::endl;
    conclusion.print(&propertyFile);
}
}
