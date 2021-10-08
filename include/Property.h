#ifndef PROPERTY_H
#define PROPERTY_H

#include <string>
#include <fstream>
#include "reluka.h"
#include "VariableManager.h"
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

        void print(); // provisório

    protected:

    private:
        std::string vnnlibFileName;
        std::ifstream vnnlibFile;
        std::string currentVnnlibLine;
        size_t currentLinePosition;
        pwl2limodsat::VariableManager *variableManager;

        unsigned inputDimension = 0, outputDimension = 0;
        std::vector<lukaFormula::Formula> premiseFormulas;
        lukaFormula::Formula conclusionFormula;
        bool propertyBuilding = false;

        enum AssertType { Undefined, Input, Output };
        enum AtomicAssertType { LessEq, GreaterEq };

        size_t nextNonSpace();
        void parseVnnlibDeclareConst();
        std::pair<AssertType,lukaFormula::Modsat> parseVnnlibAtomicAssert(AtomicAssertType atomicAssertType);
        std::pair<AssertType,lukaFormula::Modsat> parseVnnlibAssert();
        void vnnlib2property();
};
}

#endif // PROPERTY_H
