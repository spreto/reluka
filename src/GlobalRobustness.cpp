#include "GlobalRobustness.h"
#include "NeuralNetwork.h"

namespace reluka
{
GlobalRobustness::GlobalRobustness(const std::vector<unsigned>& inputNnConsideredOutputIndexes,
                                   std::string onnxFileName,
                                   size_t inputNnInputDimension,
                                   size_t inputNnOutputDimension,
                                   std::vector<pwl2limodsat::PiecewiseLinearFunction> *inputNnOutputAddresses,
                                   double inputEpsilon,
                                   pwl2limodsat::VariableManager *varMan) :
    nnConsideredOutputIndexes(inputNnConsideredOutputIndexes),
    nnInputDimension(inputNnInputDimension),
    nnOutputDimension(inputNnOutputDimension),
    nnOutputAddresses(inputNnOutputAddresses),
    variableManager(varMan)
{
    if ( nnOutputDimension != nnOutputAddresses->size() )
        throw std::invalid_argument("Pwl addresses size is not coherent.");

    epsilon = NeuralNetwork::dec2frac(inputEpsilon);

    std::string generalPropertyFileName;

    if ( onnxFileName.substr(onnxFileName.size()-5,5) == ".onnx" )
        generalPropertyFileName = onnxFileName.substr(0,onnxFileName.size()-5);
    else
        generalPropertyFileName = onnxFileName;

    if ( nnConsideredOutputIndexes.empty() )
        for ( size_t outIdx = 0; outIdx < nnOutputDimension; outIdx++ )
            nnConsideredOutputIndexes.push_back(outIdx);

    for ( size_t outIdx = 0; outIdx < nnConsideredOutputIndexes.size(); outIdx++ )
        propertyFileName.push_back(generalPropertyFileName + "_" + std::to_string(nnConsideredOutputIndexes.at(outIdx)) + ".liprop");
}

GlobalRobustness::GlobalRobustness(std::string onnxFileName,
                                   size_t inputNnInputDimension,
                                   size_t inputNnOutputDimension,
                                   std::vector<pwl2limodsat::PiecewiseLinearFunction> *inputNnOutputAddresses,
                                   double inputEpsilon,
                                   pwl2limodsat::VariableManager *varMan) :
    GlobalRobustness(std::vector<unsigned>(),
                     onnxFileName,
                     inputNnInputDimension,
                     inputNnOutputDimension,
                     inputNnOutputAddresses,
                     inputEpsilon,
                     varMan) {}

void GlobalRobustness::buildCloneRepresentations()
{
    for ( size_t i = 0; i < nnInputDimension; i++ )
        nnCloneInputInfo.push_back(variableManager->newVariable());

    pwl2limodsat::Variable curVar = variableManager->currentVariable();
    pwl2limodsat::Variable newMaxVar = pwl2limodsat::Variable(0);

    std::vector<lukaFormula::Formula> repFormClone;

    for ( pwl2limodsat::PiecewiseLinearFunction pwl : *nnOutputAddresses )
    {
        repFormClone.push_back(pwl.getLatticeFormula());
        pwl2limodsat::Variable auxMaxVar = repFormClone.back().shiftVariables(nnCloneInputInfo, curVar);
        newMaxVar = ( auxMaxVar > newMaxVar ? auxMaxVar : newMaxVar );

        for ( pwl2limodsat::RegionalLinearPiece rlp : pwl.getLinearPieceCollection() )
        {
            std::vector<lukaFormula::Formula> auxVecForm(rlp.getRepresentationModsat().Phi);

            for ( lukaFormula::Formula form : auxVecForm )
            {
                auxMaxVar = form.shiftVariables(nnCloneInputInfo, curVar);
                newMaxVar = ( auxMaxVar > newMaxVar ? auxMaxVar : newMaxVar );
            }

            cloneRepresentations.insert( cloneRepresentations.end(), auxVecForm.begin(), auxVecForm.end() );
        }
    }

    variableManager->jumpToVariable(newMaxVar);

    for ( size_t i = 0; i < nnOutputDimension; i++ )
    {
        nnOutputInfo.push_back(std::pair<pwl2limodsat::Variable,
                                         pwl2limodsat::Variable>(variableManager->newVariable(),
                                                                 variableManager->newVariable()));
        nnOutputAddresses->at(i).equivalentTo(nnOutputInfo.back().first);

        repFormClone.at(i).addEquivalence(lukaFormula::Formula(nnOutputInfo.back().second));
    }

    cloneRepresentations.insert( cloneRepresentations.end(), repFormClone.begin(), repFormClone.end() );
}

pwl2limodsat::Variable GlobalRobustness::buildEpsilonFormulas()
{
    if ( !variableManager->isThereConstant(epsilon.second) )
    {
        lukaFormula::ModsatSet msSet = pwl2limodsat::LinearPiece::defineConstant(variableManager,
                                                                                 epsilon.second);
        epsilonFormulas.insert(epsilonFormulas.end(), msSet.begin(), msSet.end());
    }

    lukaFormula::Modsat epsModsat = pwl2limodsat::LinearPiece::multiplyConstant(variableManager,
                                                                                epsilon.first,
                                                                                epsilon.second);

    epsilonFormulas.push_back(epsModsat.phi);
    epsilonFormulas.insert(epsilonFormulas.end(), epsModsat.Phi.begin(), epsModsat.Phi.end());

    pwl2limodsat::Variable firstPerturbationVariable = variableManager->newVariable();

    epsilonFormulas.push_back( lukaFormula::Formula(lukaFormula::Formula(firstPerturbationVariable),
                                                    epsModsat.phi,
                                                    Impl) );

    for ( size_t i = 1; i < nnInputDimension; i++ )
        epsilonFormulas.push_back( lukaFormula::Formula(lukaFormula::Formula(variableManager->newVariable()),
                                                        epsModsat.phi,
                                                        Impl) );

    return firstPerturbationVariable;
}

void GlobalRobustness::buildPerturbationFormulas(pwl2limodsat::Variable firstPerturbVar)
{
    for ( size_t i = 0; i < nnInputDimension; i++ )
    {
        perturbationFormulas.push_back(lukaFormula::Formula(lukaFormula::Formula(lukaFormula::Formula(lukaFormula::Formula(nnCloneInputInfo.at(i)),
                                                                                                      lukaFormula::Formula(lukaFormula::Formula(pwl2limodsat::Variable(i+1)),
                                                                                                                           lukaFormula::Formula(firstPerturbVar + (pwl2limodsat::Variable) i),
                                                                                                                           Lor),
                                                                                                      Equiv)),
                                                            lukaFormula::Formula(lukaFormula::Formula(lukaFormula::Formula(nnCloneInputInfo.at(i)),
                                                                                                      lukaFormula::Formula(lukaFormula::Formula(lukaFormula::Formula(pwl2limodsat::Variable(i+1)),
                                                                                                                                                lukaFormula::Formula(firstPerturbVar + (pwl2limodsat::Variable) i),
                                                                                                                                                Impl),
                                                                                                                           Neg),
                                                                                                      Equiv)),
                                                            Max));
    }
}

void GlobalRobustness::buildPremisseAndConclusionFormulas()
{
    if ( nnInputDimension == 1 )
    {
        if ( !variableManager->isThereConstant(2) )
            variableManager->newConstant(2);

        premisseFormulas.push_back( lukaFormula::Formula(lukaFormula::Formula(variableManager->constant(2)), // tá errado
                                                         nnOutputInfo.at(0).first,
                                                         Impl) );

        conclusionFormulas.push_back( lukaFormula::Formula(lukaFormula::Formula(variableManager->constant(2)),
                                                           nnOutputInfo.at(0).second,
                                                           Impl) );
    }
    else
    {}
}

void GlobalRobustness::buildRobustnessProperty()
{
    buildCloneRepresentations();
    buildPerturbationFormulas( buildEpsilonFormulas() );
    buildPremisseAndConclusionFormulas();

    propertyBuilding = true;
}

void GlobalRobustness::printLipropFile()
{
    if ( !propertyBuilding )
        buildRobustnessProperty();

    for ( size_t i = 0; i < propertyFileName.size(); i++ )
    {
        std::ofstream propertyFile(propertyFileName.at(i));
        propertyFile << "Cons" << std::endl << std::endl;

        for ( pwl2limodsat::PiecewiseLinearFunction pwl : *nnOutputAddresses )
        {
            for ( pwl2limodsat::RegionalLinearPiece rlp : pwl.getLinearPieceCollection() )
                rlp.printModsatSetAs(&propertyFile, "f:");

            propertyFile << "f:" << std::endl;
            pwl.getLatticeFormula().print(&propertyFile);
        }

        for ( lukaFormula::Formula cForm : cloneRepresentations )
        {
            propertyFile << "f:" << std::endl;
            cForm.print(&propertyFile);
        }

        for ( lukaFormula::Formula epsForm : epsilonFormulas )
        {
            propertyFile << "f:" << std::endl;
            epsForm.print(&propertyFile);
        }

        for ( lukaFormula::Formula pForm : perturbationFormulas )
        {
            propertyFile << "f:" << std::endl;
            pForm.print(&propertyFile);
        }

        propertyFile << "f:" << std::endl;
        premisseFormulas.at(i).print(&propertyFile);

        propertyFile << "C:" << std::endl;
        conclusionFormulas.at(i).print(&propertyFile);
    }
}
}
