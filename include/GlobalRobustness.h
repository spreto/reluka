#ifndef GLOBALROBUSTNESS_H
#define GLOBALROBUSTNESS_H

#include <string>
#include "reluka.h"
#include "VariableManager.h"
#include "PiecewiseLinearFunction.h"

namespace reluka
{
class GlobalRobustness
{
    public:
        GlobalRobustness(const std::vector<unsigned>& inputNnConsideredOutputIndexes,
                         std::string onnxFileName,
                         size_t inputNnInputDimension,
                         size_t inputNnOutputDimension,
                         std::vector<pwl2limodsat::PiecewiseLinearFunction> *inputNnOutputAddresses,
                         double inputEpsilon,
                         pwl2limodsat::VariableManager *varMan);
        GlobalRobustness(std::string onnxFileName,
                         size_t inputNnInputDimension,
                         size_t inputNnOutputDimension,
                         std::vector<pwl2limodsat::PiecewiseLinearFunction> *inputNnOutputAddresses,
                         double inputEpsilon,
                         pwl2limodsat::VariableManager *varMan);
        void buildRobustnessProperty();
        void printLipropFile();

    protected:

    private:
        std::vector<std::string> propertyFileName;

        std::vector<unsigned> nnConsideredOutputIndexes;
        size_t nnInputDimension, nnOutputDimension;
        std::vector<pwl2limodsat::PiecewiseLinearFunction> *nnOutputAddresses;
        pwl2limodsat::LinearPieceCoefficient epsilon;
        pwl2limodsat::VariableManager *variableManager;

        std::vector<pwl2limodsat::Variable> nnCloneInputInfo;
        std::vector<std::pair<pwl2limodsat::Variable,pwl2limodsat::Variable>> nnOutputInfo;

        std::vector<lukaFormula::Formula> cloneRepresentations;
        std::vector<lukaFormula::Formula> epsilonFormulas;
        std::vector<lukaFormula::Formula> perturbationFormulas;
        std::vector<lukaFormula::Formula> premisseFormulas;
        std::vector<lukaFormula::Formula> conclusionFormulas;
        bool propertyBuilding = false;

        void buildCloneRepresentations();
        pwl2limodsat::Variable buildEpsilonFormulas();
        void buildPerturbationFormulas(pwl2limodsat::Variable firstPerturbVar);
        void buildPremisseAndConclusionFormulas();
};
}

#endif // GLOBALROBUSTNESS_H
