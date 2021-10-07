#ifndef PROPERTY_H
#define PROPERTY_H

#include <string>
#include <fstream>
#include "reluka.h"
#include "Formula.h"

namespace reluka
{
class Property
{
    public:
        Property(const char* inputVnnlibFileName);
        virtual ~Property();
        void buildProperty();

    protected:

    private:
        std::string vnnlibFileName;
        std::ifstream vnnlibFile;

        unsigned inputDimension = 0, outputDimension = 0;
        std::vector<lukaFormula::Formula> assertFormulas;

        std::string currentVnnlibLine;
        size_t currentLinePosition;

        size_t nextNonSpace();
        void parseVnnlibDeclareConst();
        enum AtomicFormulaType { LessEq, GreaterEq };
        lukaFormula::Formula parseVnnlibInequality(AtomicFormulaType type);
        lukaFormula::Formula parseVnnlibAssert();
        void vnnlib2property();
};
}

#endif // PROPERTY_H
