
import os
import numpy as np
import cython as cy

cimport numpy as np

dataDir = os.path.dirname(__file__) + 'coeffsData/'
coeffsAllDict = {}
allFileNames = os.listdir(dataDir)
allFileNames = sorted(allFileNames)

with cy.wraparound(True):
    while len(allFileNames) > 0:
        currentCoeffsName = allFileNames[0][:-7]
        currentCoeffsDict = {}
        coeffsAllDict[currentCoeffsName] = currentCoeffsDict
        while len(allFileNames) > 0 and currentCoeffsName == allFileNames[0][:-7]:
            currentCoeffsDict[int(allFileNames[0][-6:-4])] = np.load(dataDir + allFileNames[0]).astype(np.double)
            allFileNames.remove(allFileNames[0])


# Actual function to get the coefficients
cdef np.ndarray getCoeffs(coeffsName, int order):
    return coeffsAllDict[coeffsName][order]



#print dataDir
#print coeffsAllDict
#print getCoeffs('coeffs_smooth', 2)
#print getCoeffs('coeffs_smooth', 10)

