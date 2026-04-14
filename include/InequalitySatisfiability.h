#ifndef INEQUALITYSATISFIABILITY_H
#define INEQUALITYSATISFIABILITY_H

#include <string>
#include <fstream>
#include "VariableManager.h"
#include "reluka.h"
#include "Formula.h"
#include "PiecewiseLinearFunction.h"
#include "NeuralNetworkModSat.h"

namespace reluka
{
class InequalitySatisfiability
{
    public:
        InequalitySatisfiability(std::string ineqsatFileName,
                                 size_t inputDim,
                                 pwl2limodsat::VariableManager *varMan);
        virtual ~InequalitySatisfiability();
        void buildIneqsatProperty(std::map<unsigned,std::pair<double,double>> originalOutputLim);
        std::map<unsigned,std::pair<double,double>> getInputLimits();
        std::vector<unsigned> getNnOutputIndexes();
        void printLiproperty(NeuralNetworkModSat *nnms);

    protected:

    private:
        std::ifstream ineqsatFile;
        std::string currentIneqsatLine;
        size_t currentLinePosition = 0;
        std::string propertyFileName;

        pwl2limodsat::VariableManager *variableManager;

        bool ineqsatParsing = false;
        bool ineqsatProperty = false;

        std::map<unsigned,std::pair<double,double>> nnInputLimits;
        std::map<unsigned,std::pair<double,double>> nnOutputLimits;
        std::vector<pwl2limodsat::Variable> nnOutputVariables;
        std::vector<unsigned> nnOutputIndexes;
        std::vector<pwl2limodsat::PiecewiseLinearFunction> *nnOutputAddresses;

        std::vector<lukaFormula::Formula> instance;

        void parseIneqsat();
};
}

#endif // INEQUALITYSATISFIABILITY_H
