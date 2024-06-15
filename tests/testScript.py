#!/var/tmp/spreto/anaconda3/bin/python
#!/usr/bin/python3

from enum import Enum

class TestMode(Enum):
    ELSE = 0
    PWL = 1
    MultiPWL = 2
    LIMODSAT = 3
    countPWLvarLayers = 4
    countPWLvarNodes = 5

PRECISION = 5
DECPRECISION_form = ".5f"
reluka_path = "../bin/Release/reluka"
yices_path = "/var/tmp/spreto/yices-smt2"
#yices_path = "yices-smt2"

import torch
import torch.onnx
import csv
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

def runPwlTest(fileName, torchModel, pwlData, outputNum):
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

        if abs(torchValue[outputNum].item() - pwlValue) < 10**-PRECISION:
            singleResult += "SUCCESS :-D | "
            statistics[0] += 1
        else:
            singleResult += "FAIL!! :-(  | "
            statistics[1] += 1

        for j in range(inputDim):
            singleResult += "x" + str(j+1) + ": {:.{}f}".format(x[j], PRECISION) + " | "
        singleResult += "| pwl: " + "{:.{}f}".format(pwlValue, PRECISION) + " | torch: " + "{:.{}f}".format(torchValue[outputNum].item(), PRECISION)

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
    os.system(reluka_path+" -onnx "+data_folder+fileName+".onnx -pwl")

    pwlData = pwlFileParser(fileName+"_0.pwl")

    runPwlTest(fileName, torchModel, pwlData, 0)

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

def runLimodsatTest(fileName, torchModel, inputDim):
    global summary
    results = []
    statistics = [0,0]

    for i in range(SINGLE_NN_TEST_NUM):
        x = []
        for j in range(inputDim):
            x.append(random.uniform(0,1))
    
        createSmt(fileName+"_0.limodsat", fileName+"_"+str(i)+".smt", inputDim, x)
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

def runRandomLimodsatTest(fileName, inputDim, hiddenDim, hiddenNum):
    torchModel = RandPwlNeuralNet(inputDim, hiddenDim, hiddenNum)
    torch.save(torchModel, data_folder+fileName+".torch")

    toOnnxInput = torch.as_tensor([0]*inputDim).float()
    torch.onnx.export(torchModel, toOnnxInput, data_folder+fileName+".onnx")

    os.system(reluka_path+" -onnx "+data_folder+fileName+".onnx -limodsat")

    runLimodsatTest(fileName, torchModel, inputDim)

def runMultiPwlTest(fileName, torchModel, pwlData):
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

def runRandomMultiPwlTest(fileName, inputDim, hiddenDim, hiddenNum, outputDim):
    torchModel = RandPwlNeuralNet(inputDim, hiddenDim, hiddenNum, outputDim)
    torch.save(torchModel, data_folder+fileName+".torch")

    toOnnxInput = torch.as_tensor([0]*inputDim).float()
    torch.onnx.export(torchModel, toOnnxInput, data_folder+fileName+".onnx")

    os.system(reluka_path+" -onnx "+data_folder+fileName+".onnx -pwl")

    for outputNum in range(outputDim):
        pwlData = pwlFileParser(fileName+"_"+str(outputNum)+".pwl")
        runPwlTest(fileName+"_"+str(outputNum), torchModel, pwlData, outputNum)

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

def runRandomPwlCountTest(fileName, inputDim, hiddenDim, hiddenNum):
    global summary

    message = [fileName]
    messageLP = []

    for config in range(SINGLE_CONFIG_TEST_NUM):
        fileNameBkp = fileName
        fileName += "_n"+str(config+1)

        torchModel = RandPwlNeuralNet(inputDim, hiddenDim, hiddenNum)
        torch.save(torchModel, data_folder+fileName+".torch")

        toOnnxInput = torch.as_tensor([0]*inputDim).float()
        torch.onnx.export(torchModel, toOnnxInput, data_folder+fileName+".onnx")
        os.system(reluka_path+" -onnx "+data_folder+fileName+".onnx -pwl -lpcount > "+data_folder+"temp")

        with open(data_folder+fileName+"_0.pwl", "r") as pwlFile:
            cnt = 0
            for line in pwlFile.readlines():
                if line.startswith("p "):
                    cnt += 1
        message.append(str(cnt))

        with open(data_folder+"temp") as tempFile:
            msgAux = tempFile.read()[6:-1]
            messageLP.append(msgAux)

        fileName = fileNameBkp

    summary.append(message+messageLP)

    os.system("rm "+data_folder+"temp")

######################################
TEST_MODE = TestMode.countPWLvarLayers

SINGLE_CONFIG_TEST_NUM = 25
SINGLE_NN_TEST_NUM = 2
MAX_INPUTS = 2
MAX_OUTPUTS = 2
MAX_NODES = 10
MAX_LAYERS = 10

# for countPWL
NUM_FIX_NODES = 5
NUM_FIX_LAYERS = 5
######################################

summary = []

#
# For each configuration of neural network with {1,...,MAX_INPUTS} inputs, one output, {1,...,MAX_NODES} nodes in each layer of {1,...,MAX_LAYERS} layers,
# compare the evaluation of SINGLE_CONFIG_TEST_NUM neural networks to the evaluation of its .pwl representation, for SINGLE_NN_TEST_NUM random tests.
#
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

#
# For each configuration of neural network with {1,...,MAX_INPUTS} inputs, one output, {1,...,MAX_NODES} nodes in each layer of {1,...,MAX_LAYERS} layers,
# compare the evaluation of SINGLE_CONFIG_TEST_NUM neural networks to the evaluation of its .limodsat representation, for SINGLE_NN_TEST_NUM random tests.
#
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

#
# For each configuration of neural network with {1,...,MAX_INPUTS} inputs, {1,...,MAX_OUTPUTS} outputs, {1,...,MAX_NODES} nodes in each layer
# of {1,...,MAX_LAYERS} layers, compare the evaluation of SINGLE_CONFIG_TEST_NUM neural networks to the evaluation of its .pwl representation,
# for SINGLE_NN_TEST_NUM random tests.
#
elif TEST_MODE is TestMode.MultiPWL:
    data_folder = "./multiPwlTestData/"
    setDataFolder()

    for inputsNum in range(MAX_INPUTS):
        for nodesNum in range(MAX_NODES):
            for layersNum in range(MAX_LAYERS):
                for outputsNum in range(MAX_OUTPUTS):
                    for config in range(SINGLE_CONFIG_TEST_NUM):
                        runRandomMultiPwlTest("test_"+str(inputsNum+1)+"_"+str(nodesNum+1)+"_"+str(layersNum+1)+"_"+str(outputsNum+1)+"_n"+str(config+1),
                                              inputsNum+1,
                                              nodesNum+1,
                                              layersNum+1,
                                              outputsNum+1)

    createSummary()

#
# For each configuration of neural network with NUM_FIX_NODES inputs, one output, NUM_FIX_NODES nodes in each layer of {1,...,MAX_LAYERS} layers,
# count the number of nonempty regions in the .pwl representation of SINGLE_CONFIG_TEST_NUM neural networks. Also, count the number of pairs of regions
# in the .pwl representation that fail to fulfill the lattice property.
#
elif TEST_MODE is TestMode.countPWLvarLayers:
    data_folder = "./pwlCountVarLayersTestData_"+str(NUM_FIX_NODES)+"n_"+str(MAX_LAYERS)+"l/"
    setDataFolder()

    message = [""]
    message.extend(range(1,SINGLE_CONFIG_TEST_NUM+1))
    message.extend(range(1,SINGLE_CONFIG_TEST_NUM+1))
    summary.append(message)

    for layersNum in range(MAX_LAYERS):
        runRandomPwlCountTest("test_"+str(NUM_FIX_NODES)+"n_"+str(layersNum+1)+"l",
                              NUM_FIX_NODES,
                              NUM_FIX_NODES,
                              layersNum+1)

    summary_file = open(data_folder+"summary.csv", "w")
    summary_writer = csv.writer(summary_file)
    for sum in summary:
        summary_writer.writerow(sum)
    summary_file.close()

#
# For each configuration of neural network with NUM_FIX_LAYERS layers, the same number {1,...,MAX_NODES} of inputs and nodes in each layer and one output,
# count the number of nonempty regions in the .pwl representation of SINGLE_CONFIG_TEST_NUM neural networks. Also, count the number of pairs of regions
# in the .pwl representation that fail to fulfill the lattice property.
#
elif TEST_MODE is TestMode.countPWLvarNodes:
    data_folder = "./pwlCountVarNodesTestData_"+str(MAX_NODES)+"n_"+str(NUM_FIX_LAYERS)+"l/"
    setDataFolder()

    message = [""]
    message.extend(range(1,SINGLE_CONFIG_TEST_NUM+1))
    message.extend(range(1,SINGLE_CONFIG_TEST_NUM+1))
    summary.append(message)

    for nodesNum in range(MAX_NODES):
        runRandomPwlCountTest("test_"+str(nodesNum+1)+"n_"+str(NUM_FIX_LAYERS)+"l",
                              nodesNum+1,
                              nodesNum+1,
                              NUM_FIX_LAYERS)

    summary_file = open(data_folder+"summary.csv", "w")
    summary_writer = csv.writer(summary_file)
    for sum in summary:
        summary_writer.writerow(sum)
    summary_file.close()

#
# Something else.
#
elif TEST_MODE is TestMode.ELSE:
    data_folder = "./elseTest/"
    setDataFolder()

#   something else here
