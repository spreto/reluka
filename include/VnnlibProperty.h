#ifndef VNNLIBPROPERTY_H
#define VNNLIBPROPERTY_H

#include <string>
#include <fstream>
#include "reluka.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"
#include "Formula.h"

namespace reluka
{
class VnnlibProperty
{
    public:
        VnnlibProperty(std::string inputVnnlibFileName,
                       pwl2limodsat::VariableManager *varMan);
        virtual ~VnnlibProperty();
        void buildVnnlibProperty();
        void setOutputAddresses(std::vector<pwl2limodsat::PiecewiseLinearFunction> *pwlAddress);
        std::vector<unsigned> getNnOutputIndexes();
        pwl2limodsat::Variable getVariable(unsigned nnOutputIdx);
        void printLipropFile();

    protected:

    private:
        std::string vnnlibFileName;
        std::ifstream vnnlibFile;
        std::string currentVnnlibLine;
        size_t currentLinePosition = 0;
        std::string propertyFileName;

        pwl2limodsat::VariableManager *variableManager;

        unsigned nnInputDimension = 0, nnOutputDimension = 0;
        std::map<unsigned,pwl2limodsat::Variable> nnOutputInfo;
        std::vector<pwl2limodsat::PiecewiseLinearFunction> *nnOutputAddresses;
        std::vector<lukaFormula::Formula> premiseFormulas;
        lukaFormula::Formula conclusionFormula;
        bool propertyBuilding = false;

        enum AssertType { Undefined, Input, Output };
        enum AtomicAssertType { LessEq, GreaterEq };

        bool nextNonSpace();
        void parseVnnlibDeclareConst();
        std::pair<AssertType,lukaFormula::Modsat> parseVnnlibAtomicAssert(AtomicAssertType atomicAssertType);
        std::pair<AssertType,lukaFormula::Modsat> parseVnnlibAssert();
        void readDeclarations();
        void vnnlib2property();
};
}

#endif // VNNLIBPROPERTY_H
