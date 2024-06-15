/*
    The code in this file may be found in
    http://github.com/spreto/pwl2limodsat
    and is available under the following license.

    MIT License

    Copyright (c) 2021 Sandro Preto

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef FORMULA_H
#define FORMULA_H

#include <vector>
#include <tuple>
#include <fstream>
#include "pwl2limodsat.h"

namespace lukaFormula
{
enum LogicalSymbol { Neg, Lor, Land, Equiv, Impl, Max, Min };

typedef int Literal;
typedef unsigned UnitIndex;

typedef std::vector<Literal> Clause;
typedef std::pair<UnitIndex,Clause> UnitClause;
typedef std::pair<UnitIndex,UnitIndex> Negation;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> BinaryOperation;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> LDisjunction;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> LConjunction;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> Equivalence;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> Implication;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> Maximum;
typedef std::tuple<UnitIndex,UnitIndex,UnitIndex> Minimum;

class Formula
{
    public:
        Formula();
        Formula(const Clause& clau);
        Formula(Literal lit);
        Formula(pwl2limodsat::Variable var);
        Formula(const Formula& form, LogicalSymbol unSym);
        Formula(const Formula& form1, const Formula& form2, LogicalSymbol binSym);
        Formula(std::vector<UnitClause> unitClausesInput,
                std::vector<Negation> negationsInput,
                std::vector<LDisjunction> lDisjunctionsInput,
                std::vector<LConjunction> lConjunctionsInput,
                std::vector<Equivalence> equivalencesInput,
                std::vector<Implication> implicationsInput,
                std::vector<Maximum> maximumsInput,
                std::vector<Minimum> minimumsInput);

        bool isEmpty() const { return emptyFormula; }

        void negateFormula();
        void addLukaDisjunction(const Formula& form);
        void addLukaConjunction(const Formula& form);
        void addEquivalence(const Formula& form);
        void addImplication(const Formula& form);
        void addMaximum(const Formula& form);
        void addMinimum(const Formula& form);
        pwl2limodsat::Variable shiftVariables(std::vector<pwl2limodsat::Variable> newInputs,
                                              pwl2limodsat::Variable byVar);
        unsigned getUnitCounter() const { return unitCounter; }
        std::vector<UnitClause> getUnitClauses() const { return unitClauses; }
        std::vector<Negation> getNegations() const { return negations; }
        std::vector<LDisjunction> getLDisjunctions() const { return lDisjunctions; }
        std::vector<LDisjunction> getLConjunctions() const { return lConjunctions; }
        std::vector<Equivalence> getEquivalences() const { return equivalences; }
        std::vector<Implication> getImplications() const { return implications; }
        std::vector<Maximum> getMaximums() const { return maximums; }
        std::vector<Minimum> getMinimums() const { return minimums; }

        void print(std::ofstream *output);

    private:
        bool emptyFormula = false;
        UnitIndex unitCounter = 0;
        std::vector<UnitClause> unitClauses;
        std::vector<Negation> negations;
        std::vector<LDisjunction> lDisjunctions;
        std::vector<LConjunction> lConjunctions;
        std::vector<Equivalence> equivalences;
        std::vector<Implication> implications;
        std::vector<Maximum> maximums;
        std::vector<Minimum> minimums;

        void addUnits(const Formula& form);
        void addBinaryOperation(const Formula& form, LogicalSymbol binSym);
};

typedef std::vector<Formula> ModsatSet;

struct Modsat
{
    Formula phi;
    ModsatSet Phi;
};
}

#endif // FORMULA_H
