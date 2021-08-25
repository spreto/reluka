#!/usr/bin/python3.8

from enum import Enum

class TestMode(Enum):
    ELSE = 0
    PWL = 1

TEST_MODE = TestMode.PWL
PRECISION = 5
reluka_path = "../bin/Release/reluka"

import torch
import torch.onnx
import os
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

############################################################################

summary = []

if TEST_MODE is TestMode.PWL:
    SINGLE_CONFIG_TEST_NUM = 5
    SINGLE_NN_TEST_NUM = 500
    MAX_INPUTS = 2
    MAX_NODES = 2
    MAX_LAYERS = 2

    data_folder = "./pwlTestData/"
    setDataFolder()

    for inputsNum in range(MAX_INPUTS):
        for nodesNum in range(MAX_NODES):
            for layersNum in range(MAX_LAYERS):
                for config in range(SINGLE_CONFIG_TEST_NUM):
                    runRandomPwlTest("test_"+str(inputsNum+1)+"_"+str(nodesNum+1)+"_"+str(layersNum+1)+"_n"+str(config+1), inputsNum+1, nodesNum+1, layersNum+1)

    createSummary()

elif TEST_MODE is TestMode.ELSE:
    data_folder = "./"
    print("Not implemented.")
