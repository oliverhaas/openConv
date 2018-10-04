
import os
import numpy as np


dataDir = os.path.dirname(__file__) + '/coeffsData/'

coeffsAllDict = {}
allFileNames = os.listdir(dataDir)
allFileNames = sorted(allFileNames)

while len(allFileNames) > 0:
    currentCoeffsName = allFileNames[0][:-7]
    currentCoeffsDict = {}
    coeffsAllDict[currentCoeffsName] = currentCoeffsDict
    while len(allFileNames) > 0 and currentCoeffsName == allFileNames[0][:-7]:
        currentCoeffsDict[int(allFileNames[0][-6:-4])] = np.load(dataDir + allFileNames[0]).astype(np.double)
        allFileNames.remove(allFileNames[0])


# Actual function to get the coefficients
def getCoeffs(coeffsName, order):
    return coeffsAllDict[coeffsName][order]
