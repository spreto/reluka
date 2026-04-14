#include "InequalitySatisfiability.h"
#include "LinearPiece.h"
#include "NeuralNetwork.h"

#include <iostream>

namespace reluka
{
InequalitySatisfiability::InequalitySatisfiability(std::string ineqsatFileName,
                                                   size_t inputDim,
                                                   pwl2limodsat::VariableManager *varMan) :
                                                   variableManager(varMan)
{
    ineqsatFile.open(ineqsatFileName);

    if ( !ineqsatFile.is_open() )
        throw std::invalid_argument("Unable to open inequality constraints file.");

    if ( ineqsatFile.eof() )
        throw std::invalid_argument("Not in standard inequality constraints file format.");

    if ( ineqsatFileName.substr(ineqsatFileName.size()-8,8) == ".ineqsat" )
        propertyFileName = ineqsatFileName.substr(0,ineqsatFileName.size()-8);
    else
        propertyFileName = ineqsatFileName;

    propertyFileName.append(".liprop");

    variableManager->setDimension(inputDim);
}

InequalitySatisfiability::~InequalitySatisfiability()
{
    ineqsatFile.close();
}

void InequalitySatisfiability::parseIneqsat()
{
    while ( !ineqsatFile.eof() )
    {
        getline(ineqsatFile, currentIneqsatLine);
        currentLinePosition = 0;

        if ( currentIneqsatLine.compare(currentLinePosition, 1, "x") == 0 )
        {
            currentLinePosition++;
            size_t blockLenght = currentIneqsatLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned inputNum = stoi(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqsatLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            double inputMin = stod(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqsatLine.size() - currentLinePosition;
            double inputMax = stod(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            nnInputLimits[inputNum] = std::pair<double,double>(inputMin,inputMax);
        }
        else if ( currentIneqsatLine.compare(currentLinePosition, 1, "y") == 0 )
        {
            currentLinePosition++;
            size_t blockLenght = currentIneqsatLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            unsigned outputNum = stoi(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqsatLine.find_first_of(" ", currentLinePosition) - currentLinePosition;
            double outputMin = stod(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            currentLinePosition += blockLenght+1;
            blockLenght = currentIneqsatLine.size() - currentLinePosition;
            double outputMax = stod(currentIneqsatLine.substr(currentLinePosition, blockLenght));
            nnOutputLimits[outputNum] = std::pair<double,double>(outputMin,outputMax);
            nnOutputIndexes.push_back(outputNum);
        }
    }

    ineqsatParsing = true;
}

void InequalitySatisfiability::buildIneqsatProperty( std::map<unsigned,std::pair<double,double>> originalOutputLim )
{
    if ( !ineqsatParsing )
        parseIneqsat();

    for ( auto& x : nnOutputIndexes )
        nnOutputVariables.push_back(variableManager->newVariable());

    for ( auto& lim : nnOutputLimits )
    {

    std::cerr << "lim.second: (" << lim.second.first << ", " << lim.second.second << ")\n";
std::cerr << "originalOutputLim: ("
          << originalOutputLim[lim.first].first << ", "
          << originalOutputLim[lim.first].second << ")\n";



        lim.second = std::pair<double,double>( (lim.second.first - originalOutputLim[lim.first].first) /
                                               (originalOutputLim[lim.first].second - originalOutputLim[lim.first].first),
                                               (lim.second.second - originalOutputLim[lim.first].first) /
                                               (originalOutputLim[lim.first].second - originalOutputLim[lim.first].first) );

        pwl2limodsat::LinearPieceCoefficient leftLim = reluka::NeuralNetwork::dec2frac(lim.second.first);
        pwl2limodsat::LinearPieceCoefficient rightLim = reluka::NeuralNetwork::dec2frac(lim.second.second);

        if ( !variableManager->isThereConstant(leftLim.second) )
            instance.push_back(pwl2limodsat::LinearPiece::defineConstant(variableManager, leftLim.second).back());
        lukaFormula::Modsat auxModsat = pwl2limodsat::LinearPiece::multiplyConstant(variableManager, leftLim.first, leftLim.second);
        instance.insert(instance.end(), auxModsat.Phi.begin(), auxModsat.Phi.end());
        auxModsat.phi.addImplication(lukaFormula::Formula(nnOutputVariables[lim.first]));
        instance.push_back(auxModsat.phi);

        if ( !variableManager->isThereConstant(rightLim.second) )
            instance.push_back(pwl2limodsat::LinearPiece::defineConstant(variableManager, rightLim.second).back());
        auxModsat = pwl2limodsat::LinearPiece::multiplyConstant(variableManager, rightLim.first, rightLim.second);
        instance.insert(instance.end(), auxModsat.Phi.begin(), auxModsat.Phi.end());
        lukaFormula::Formula auxForm(nnOutputVariables[lim.first]);
        auxForm.addImplication(auxModsat.phi);
        instance.push_back(auxForm);
    }
/*
    for ( minimais )

    for ( maximais )
*/

    ineqsatProperty = true;
}

std::map<unsigned,std::pair<double,double>> InequalitySatisfiability::getInputLimits()
{
    if ( !ineqsatParsing )
        parseIneqsat();

    return nnInputLimits;
}

std::vector<unsigned> InequalitySatisfiability::getNnOutputIndexes()
{
    if ( !ineqsatParsing )
        parseIneqsat();

    return nnOutputIndexes;
}

void InequalitySatisfiability::printLiproperty(NeuralNetworkModSat *nnms)
{
    if ( !ineqsatProperty )
        buildIneqsatProperty(nnms->getOriginalOutputLim());

    std::ofstream propertyFile(propertyFileName);
    propertyFile << "Sat" << std::endl << std::endl;

    nnms->printNNmodsat(&propertyFile, nnOutputVariables);

    for ( lukaFormula::Formula form : instance )
    {
        propertyFile << "f:" << std::endl;
        form.print(&propertyFile);
    }
}
}
