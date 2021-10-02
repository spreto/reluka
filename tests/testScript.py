#!/usr/bin/python3.8

from enum import Enum

class TestMode(Enum):
    ELSE = 0
    PWL = 1
    LIMODSAT = 2

PRECISION = 5
DECPRECISION_form = ".5f"
reluka_path = "../bin/Release/reluka"
yices_path = "yices-smt2"

import torch
import torch.onnx
import os
import sys
import subprocess
import random
from randNeuralNet import *

def setDataFolder():
    if os.path.exists(data_folder):
        os.system("rm -fr "+data_folder)

    os.system("mkdir "+data_folder)

def evaluatePwl(pwlData, evalValues):
    boundaryIndex = -1
    activeRegion = False

    dimension = len(pwlData[1][0]) - 1

    while activeRegion == False:
        boundaryIndex += 1
        activeRegion = True
        for bound in pwlData[2][boundaryIndex]:
            phi = pwlData[1][bound[1]][0]
            for i in range(dimension):
                phi += evalValues[i]*pwlData[1][bound[1]][i+1]
            if not ((phi >= 0 and bound[0] == 'g') or (phi <= 0 and bound[0] == 'l')):
                activeRegion = False

    evaluation = pwlData[0][boundaryIndex][0]/pwlData[0][boundaryIndex][1]
    for val in range(2,2*(dimension+1),2):
        evaluation += (pwlData[0][boundaryIndex][val]/pwlData[0][boundaryIndex][val+1])*evalValues[int(val/2)-1]

    return evaluation

def pwlFileParser(pwlFileName):
    lpCoefficients = []
    boundPrototypes = []
    lpBoundaries = []

    pwlFile = open(data_folder+pwlFileName, "r")

    for line in pwlFile:
        if "b " == line[0:2]:
            boundProt = []
            bPos = 2

            while bPos < len(line):
                ePos = line.find(" ", bPos) if line.find(" ", bPos) > 0 else len(line)
                boundProt.append(float(line[bPos:ePos]))
                bPos = ePos + 1

            boundPrototypes.append(boundProt)

        elif "p " == line[0:2]:
            if len(lpCoefficients):
                lpBoundaries.append(lpBound)
            lpBound = []

            lpCoeff = []
            bPos = 2

            while bPos < len(line):
                ePos = line.find(" ", bPos) if line.find(" ", bPos) > 0 else len(line)
                lpCoeff.append(int(line[bPos:ePos]))
                bPos = ePos + 1

            lpCoefficients.append(lpCoeff)

        elif "g " == line[0:2]:
            lpBound.append(['g', int(line[2:])-1])

        elif "l " == line[0:2]:
            lpBound.append(['l', int(line[2:])-1])

    lpBoundaries.append(lpBound)

    pwlFile.close()

    pwlData = [lpCoefficients, boundPrototypes, lpBoundaries]

    return pwlData

def runPwlTest(fileName, torchModel, pwlData):
    global summary
    results = []
    statistics = [0,0]

    inputDim = len(pwlData[1][0]) - 1

    for i in range(SINGLE_NN_TEST_NUM):
        x = []

        for j in range(inputDim):
            x.append(random.uniform(0,1))
    
        torchValue = torchModel(torch.as_tensor(x).float())
        pwlValue = evaluatePwl(pwlData, x)

        singleResult = "{:3d}".format(i+1) + " | "

        if abs(torchValue.item() - pwlValue) < 10**-PRECISION:
            singleResult += "SUCCESS :-D | "
            statistics[0] += 1
        else:
            singleResult += "FAIL!! :-(  | "
            statistics[1] += 1

        for j in range(inputDim):
            singleResult += "x" + str(j+1) + ": {:.{}f}".format(x[j], PRECISION) + " | "
        singleResult += "| pwl: " + "{:.{}f}".format(pwlValue, PRECISION) + " | torch: " + "{:.{}f}".format(torchValue.item(), PRECISION)

        results.append(singleResult)

    resultsFile = open(data_folder+fileName+".res", "w")

    message = fileName + ": "
    if not statistics[1]:
        message += "PASSED ALL EVALUATIONS!!!"
    elif not statistics[0]:
        message += "FAILED all evaluations :("
    else:
        message += "PASSED: " + str(statistics[0]) + " | failed: " + str(statistics[1])

    resultsFile.write(message+"\n\n")
    print(message)
    summary.append(message)

    for res in range(len(results)):
        resultsFile.write(results[res])
        if res != len(results)-1:
            resultsFile.write("\n")

    resultsFile.close()

def runRandomPwlTest(fileName, inputDim, hiddenDim, hiddenNum):
    torchModel = RandPwlNeuralNet(inputDim, hiddenDim, hiddenNum)
    torch.save(torchModel, data_folder+fileName+".torch")

    toOnnxInput = torch.as_tensor([0]*inputDim).float()
    torch.onnx.export(torchModel, toOnnxInput, data_folder+fileName+".onnx")
    os.system(reluka_path+" "+data_folder+fileName+".onnx")

    pwlData = pwlFileParser(fileName+".pwl")

    runPwlTest(fileName, torchModel, pwlData)

def createSmt(fileName, smtFileName, dimension, values):
    formula = []
    maxvar = dimension
    phi = False
    smt_aux = []

    out_file = open(data_folder+fileName, "r")

    for line in out_file:
        if "Unit" == line[0:4]:
            if ":: Clause" == line[line.find("::"):line.find("::")+9]:
                linepos_begin = line.find("::")+18
                linepos_end = linepos_begin+line[linepos_begin:].find(" ")

                if int(line[linepos_begin:linepos_end]) > 0:
                    clau = "X"+line[linepos_begin : linepos_end]
                    maxvar = max([maxvar,int(line[linepos_begin : linepos_end])])
                else:
                    clau = "(neg "+"X"+line[linepos_begin+1:linepos_end]+")"
                    maxvar = max([maxvar,int(line[linepos_begin+1:linepos_end])])

                while linepos_end+1 < len(line)-1:
                    linepos_begin = linepos_end+1
                    linepos_end = linepos_begin+line[linepos_begin:].find(" ")

                    if int(line[linepos_begin:linepos_end]) > 0:
                        clau = "(sdis "+clau+" X"+line[linepos_begin : linepos_end]+")"
                        maxvar = max([maxvar,int(line[linepos_begin : linepos_end])])
                    else:
                        clau = "(sdis "+clau+" (neg X"+line[linepos_begin+1 : linepos_end]+"))"
                        maxvar = max([maxvar,int(line[linepos_begin+1 : linepos_end])])

                formula.append(clau)

            elif ":: Negation" == line[line.find("::"):line.find("::")+11]:
                linepos_begin = line.find("::")+18
                linepos_end = len(line)-1

                formula.append("(neg "+formula[int(line[linepos_begin : linepos_end])-1]+")")

            elif ":: Implication" == line[line.find("::"):line.find("::")+14]:
                linepos_begin = line.find("::")+18
                linepos_end = linepos_begin+line[linepos_begin:].find(" ")
                linepos_begin2 = linepos_end+1
                linepos_end2 = len(line)-1

                formula.append("(impl "+formula[int(line[linepos_begin:linepos_end])-1]+" "+formula[int(line[linepos_begin2:linepos_end2])-1]+")")

            elif ":: Equivalence" == line[line.find("::"):line.find("::")+14]:
                linepos_begin = line.find("::")+18
                linepos_end = linepos_begin+line[linepos_begin:].find(" ")
                linepos_begin2 = linepos_end+1
                linepos_end2 = len(line)-1

                formula.append("(equiv "+formula[int(line[linepos_begin:linepos_end])-1]+" "+formula[int(line[linepos_begin2:linepos_end2])-1]+")")

            elif ":: Minimum" == line[line.find("::"):line.find("::")+10]:
                linepos_begin = line.find("::")+18
                linepos_end = linepos_begin+line[linepos_begin:].find(" ")
                linepos_begin2 = linepos_end+1
                linepos_end2 = len(line)-1

                formula.append("(min "+formula[int(line[linepos_begin:linepos_end])-1]+" "+formula[int(line[linepos_begin2:linepos_end2])-1]+")")

            elif ":: Maximum" == line[line.find("::"):line.find("::")+10]:
                linepos_begin = line.find("::")+18
                linepos_end = linepos_begin+line[linepos_begin:].find(" ")
                linepos_begin2 = linepos_end+1
                linepos_end2 = len(line)-1

                formula.append("(max "+formula[int(line[linepos_begin:linepos_end])-1]+" "+formula[int(line[linepos_begin2:linepos_end2])-1]+")")

        else:
            if formula:
                if not phi:
                    smt_aux.append("(assert (= phi "+formula[len(formula)-1]+"))")
                    phi = True
                else:
                    smt_aux.append("(assert (= "+formula[len(formula)-1]+" 1))")

                formula = []

    if formula:
        smt_aux.append("(assert (= "+formula[len(formula)-1]+" 1))")

    out_file.close()

    smtFile = open(data_folder+smtFileName, "w")

    smtFile.write("(set-logic QF_LRA)"+"\n")
    smtFile.write("(define-fun min ((x Real) (y Real)) Real(ite (> x y) y x))"+"\n")
    smtFile.write("(define-fun max ((x Real) (y Real)) Real(ite (> x y) x y))"+"\n")
    smtFile.write("(define-fun sdis ((x Real) (y Real)) Real(min 1 (+ x y)))"+"\n")
    smtFile.write("(define-fun scon ((x Real) (y Real)) Real(max 0 (- (+ x y) 1)))"+"\n")
    smtFile.write("(define-fun wdis ((x Real) (y Real)) Real(max x y))"+"\n")
    smtFile.write("(define-fun wcon ((x Real) (y Real)) Real(min y x))"+"\n")
    smtFile.write("(define-fun neg ((x Real)) Real(- 1 x))"+"\n")
    smtFile.write("(define-fun impl ((x Real) (y Real)) Real(min 1 (- (+ 1 y) x)))"+"\n")
    smtFile.write("(define-fun equiv ((x Real) (y Real)) Real(- 1 (max (- x y) (- y x))))"+"\n")
    smtFile.write("\n")
    smtFile.write("(declare-fun phi () Real)"+"\n")

    for var in range(1,maxvar+1):
        smtFile.write("(declare-fun X"+str(var)+" () Real)"+"\n")

    smtFile.write("\n")

    for var in range(dimension+1,maxvar+1):
        smtFile.write("(assert (>= X"+str(var)+" 0))"+"\n")
        smtFile.write("(assert (<= X"+str(var)+" 1))"+"\n")

    smtFile.write("\n")

    for string in smt_aux:
        smtFile.write(string+"\n")

    smtFile.write("\n")

    for val in range(len(values)):
        smtFile.write("(assert (= X"+str(val+1)+" "+format(values[val], DECPRECISION_form)+"))"+"\n")

    smtFile.write("\n")
    smtFile.write("(check-sat)")
    smtFile.write("\n")
    smtFile.write("(get-value (phi))")

    smtFile.close()

def evaluateSmt(smtFileName):
    smtOut = subprocess.check_output([yices_path, data_folder+smtFileName]).decode(sys.stdout.encoding)

    if smtOut[smtOut.find("phi")+4 : smtOut.find("phi")+6] != "(/":
        evaluation = int(smtOut[smtOut.find("phi")+4 : smtOut.find(")")])
    else:
        beginpos = smtOut.find("phi")+7
        endpos = beginpos+smtOut[beginpos:].find(" ")
        beginpos2 = endpos+1
        endpos2 = beginpos2+smtOut[beginpos2:].find(")")
        evaluation = int(smtOut[beginpos : endpos]) / int(smtOut[beginpos2 : endpos2])

    return evaluation

def runLimodsatTest(fileName, torchModel, inputDim, latticePropertyMessage):
    global summary
    results = []
    statistics = [0,0]

    for i in range(SINGLE_NN_TEST_NUM):
        x = []
        for j in range(inputDim):
            x.append(random.uniform(0,1))
    
        createSmt(fileName+".limodsat", fileName+"_"+str(i)+".smt", inputDim, x)
        torchValue = torchModel(torch.as_tensor(x).float())
        modsatValue = evaluateSmt(fileName+"_"+str(i)+".smt")
        os.system("rm "+data_folder+fileName+"_"+str(i)+".smt")

        singleResult = "{:3d}".format(i+1) + " | "

        if abs(torchValue.item() - modsatValue) < 10**-PRECISION:
            singleResult += "SUCCESS :-D | "
            statistics[0] += 1
        else:
            singleResult += "FAIL!! :-(  | "
            statistics[1] += 1

        for j in range(inputDim):
            singleResult += "x" + str(j+1) + ": {:.{}f}".format(x[j], PRECISION) + " | "
        singleResult += "| limodsat: " + "{:.{}f}".format(modsatValue, PRECISION) + " | torch: " + "{:.{}f}".format(torchValue.item(), PRECISION)

        results.append(singleResult)

    resultsFile = open(data_folder+fileName+".res", "w")

    message = fileName + ": " + latticePropertyMessage + " :: "
    if not statistics[1]:
        message += "PASSED ALL EVALUATIONS!!!"
    elif not statistics[0]:
        message += "FAILED all evaluations :("
    else:
        message += "PASSED: " + str(statistics[0]) + " | failed: " + str(statistics[1])

    resultsFile.write(message+"\n\n")
    print(message)
    summary.append(message)

    for res in range(len(results)):
        resultsFile.write(results[res])
        if res != len(results)-1:
            resultsFile.write("\n")

    resultsFile.close()

def runRandomLimodsatTest(fileName, inputDim, hiddenDim, hiddenNum):
    torchModel = RandPwlNeuralNet(inputDim, hiddenDim, hiddenNum)
    torch.save(torchModel, data_folder+fileName+".torch")

    toOnnxInput = torch.as_tensor([0]*inputDim).float()
    torch.onnx.export(torchModel, toOnnxInput, data_folder+fileName+".onnx")

    latticePropertyMessage = subprocess.check_output([reluka_path, data_folder+fileName+".onnx"]).decode(sys.stdout.encoding).rstrip("\n")

    runLimodsatTest(fileName, torchModel, inputDim, latticePropertyMessage)

def createSummary():
    global summary

    final_analysis = "PASSED ALL TESTS!!! :D"

    for sum in summary:
        if sum[-3:] != "!!!":
            final_analysis = "Failed in at least one test... :("

    summary_file = open(data_folder+"summary.res", "w")

    summary_file.write(final_analysis)
    summary_file.write("\n\n")

    for sum in summary:
        summary_file.write(sum+"\n")

    summary_file.close()

#############################
TEST_MODE = TestMode.LIMODSAT

SINGLE_CONFIG_TEST_NUM = 5
SINGLE_NN_TEST_NUM = 10
MAX_INPUTS = 5
MAX_NODES = 10
MAX_LAYERS = 6
#############################

summary = []

if TEST_MODE is TestMode.PWL:
    data_folder = "./pwlTestData/"
    setDataFolder()

    for inputsNum in range(MAX_INPUTS):
        for nodesNum in range(MAX_NODES):
            for layersNum in range(MAX_LAYERS):
                for config in range(SINGLE_CONFIG_TEST_NUM):
                    runRandomPwlTest("test_"+str(inputsNum+1)+"_"+str(nodesNum+1)+"_"+str(layersNum+1)+"_n"+str(config+1),
                                     inputsNum+1,
                                     nodesNum+1,
                                     layersNum+1)

    createSummary()

elif TEST_MODE is TestMode.LIMODSAT:
    data_folder = "./limodsatTestData/"
    setDataFolder()

    for inputsNum in range(MAX_INPUTS):
        for nodesNum in range(MAX_NODES):
            for layersNum in range(MAX_LAYERS):
                for config in range(SINGLE_CONFIG_TEST_NUM):
                    runRandomLimodsatTest("test_"+str(inputsNum+1)+"_"+str(nodesNum+1)+"_"+str(layersNum+1)+"_n"+str(config+1),
                                          inputsNum+1,
                                          nodesNum+1,
                                          layersNum+1)

    createSummary()

elif TEST_MODE is TestMode.ELSE:
    print("Not implemented.")
