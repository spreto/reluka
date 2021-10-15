#ifndef PROPERTY_H
#define PROPERTY_H

#include <string>
#include <fstream>
#include "reluka.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"
#include "Formula.h"

namespace reluka
{
class Property
{
    public:
        Property(const char* inputVnnlibFileName,
                 pwl2limodsat::VariableManager *varMan);
        virtual ~Property();
        void buildProperty();
        void setOutputAddress(pwl2limodsat::PiecewiseLinearFunction *pwlAddress);
        void printLipropFile();

    protected:

    private:
        std::string vnnlibFileName;
        std::ifstream vnnlibFile;
        std::string currentVnnlibLine;
        size_t currentLinePosition = 0;
        std::string propertyFileName;

        pwl2limodsat::VariableManager *variableManager;

        unsigned inputDimension = 0, outputDimension = 0;
        std::map<unsigned,pwl2limodsat::Variable> outputInfo;
        std::vector<pwl2limodsat::PiecewiseLinearFunction*> outputAddresses;
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

#endif // PROPERTY_H
