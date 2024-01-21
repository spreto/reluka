#ifndef INEQUALITYCONSTRAINTS_H
#define INEQUALITYCONSTRAINTS_H

#include <string>
#include <fstream>
#include "VariableManager.h"
#include "reluka.h"
#include "Formula.h"
#include "PiecewiseLinearFunction.h"

enum outputLimitsType { LessEq, GreaterEq, Both };

namespace reluka
{
class InequalityConstraints
{
    public:
        InequalityConstraints(std::string vnnlibFileName,
                              pwl2limodsat::VariableManager *varMan);
        virtual ~InequalityConstraints();
        void buildIneqconsProperty();
        std::map<unsigned,std::pair<double,double>> getInputLimits() { return nnInputLimits; }
        std::map<unsigned,std::tuple<outputLimitsType,double,double>> getOutputLimits() { return nnOutputLimits; }
        std::vector<unsigned> getNnOutputIndexes() { return nnOutputIndexes; }
        void setOutputAddresses(std::vector<pwl2limodsat::PiecewiseLinearFunction> *pwlAddresses);

    protected:

    private:
        std::ifstream ineqconsFile;
        std::string currentIneqconsLine;
        size_t currentLinePosition = 0;
        std::string propertyFileName;

        pwl2limodsat::VariableManager *variableManager;

        std::map<unsigned,std::pair<double,double>> nnInputLimits;
        std::map<unsigned,std::tuple<outputLimitsType,double,double>> nnOutputLimits;
        std::map<unsigned,pwl2limodsat::Variable> nnOutputVariables;
        std::vector<unsigned> nnOutputIndexes;
        std::vector<pwl2limodsat::PiecewiseLinearFunction> *nnOutputAddresses;

        std::vector<lukaFormula::Formula> premises;
        lukaFormula::Formula conclusion;

        void parseIneqcons();
};
}

#endif // INEQUALITYCONSTRAINTS_H
