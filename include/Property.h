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

    protected:

    private:
        std::string vnnlibFileName;
        std::ifstream vnnlibFile;
        pwl2limodsat::VariableManager *variableManager;

        unsigned inputDimension = 0, outputDimension = 0;
        std::vector<lukaFormula::Formula> assertFormulas;
        bool propertyBuilding = false;

        std::string currentVnnlibLine;
        size_t currentLinePosition;

        size_t nextNonSpace();
        void parseVnnlibDeclareConst();
        enum AtomicAssertType { LessEq, GreaterEq };
        lukaFormula::Modsat parseVnnlibInequality(AtomicAssertType type);
        lukaFormula::Formula parseVnnlibAssert();
        void vnnlib2property();
};
}

#endif // PROPERTY_H
