############################################################################################################################################
# Simple example which calculates the convolution of two Guassians.
# Results are compared with the analytical solution. Mostly default parameters are used.
############################################################################################################################################


import openConv as oc
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as mpl


# Parameters and input data
order = 1
orderM1Half = max(int((order-1)/2),0)

nData = 2009
xMaxData = 8.
sigData = 0.5
stepSize = xMaxData/(nData-1)
xData = np.linspace(-orderM1Half*stepSize, orderM1Half*stepSize+xMaxData, nData+2*orderM1Half)
data = xData*np.exp(-xData/sigData)

# Kernel
lamKernel = 0.01
def kern(xx):
    return np.exp(-xx/lamKernel)# + 1.e4*np.exp(-2.*xx/lamKernel)
nKernel = 3250


## Parameters and output result
#sigResult = np.sqrt(sigData**2+sigKernel**2)
nResult = nData+nKernel-1   # Can be chosen arbitrary though
xResult = np.linspace(0., (nResult-1)*stepSize, nResult)

xKernel = np.linspace(0., (nResult+nData-2)*stepSize, nResult+nData-1)
kernel = kern(xKernel)

############################################################################################################################################
# Create convolution object, which does all precomputation possible without knowing the exact 
# data. This way it's much faster if repeated convolutions with the same kernel are done.
convObj = oc.Conv(nData, 1, kern, None, 2, stepSize, nResult, method = 0, order = order)    
result = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

convObj = oc.Conv(nData, 1, kern, None, 2, stepSize, nResult, method = 1, order = order)    
result2 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

convObj = oc.Conv(nData, 1, kern, None, 2, stepSize, nResult, method = 2, order = order)    
result3 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

convObj = oc.Conv(nData, 1, kern, None, 2, stepSize, nResult, method = 3, order = order)    
result4 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)


mpl.figure()
#mpl.semilogy(xData, data, label='data')
#mpl.semilogy(xKernel, kernel, label='kernel')
mpl.semilogy(xResult/stepSize, np.abs(result2), marker='o', label='fft')
mpl.semilogy(xResult/stepSize, np.abs(result3), marker='o', label='fmmCheb')
mpl.semilogy(xResult/stepSize, np.abs(result4), marker='o', label='fmmExpCheb')
mpl.semilogy(xResult/stepSize, np.abs(result), marker='o', label='trap')
#mpl.semilogy(xResult, resultAna)
mpl.legend()


mpl.figure()
mpl.semilogy(xResult[:-1]/stepSize, np.clip(np.abs((result3[:-1]-result[:-1])/result[:-1]),1.e-17,10.), marker='o', label='fmmCheb')
mpl.semilogy(xResult[:-1]/stepSize, np.clip(np.abs((result4[:-1]-result[:-1])/result[:-1]),1.e-17,10.), marker='o', label='fmmExpCheb')
mpl.semilogy(xResult[:-1]/stepSize, np.clip(np.abs((result2[:-1]-result[:-1])/result[:-1]),1.e-17,10.), marker='o', label='fft')
mpl.legend()
mpl.show()

## Plotting
#fig, axarr = mpl.subplots(2, 1, sharex=True)

#axarr[0].plot(xx, dataOutAna, 'r--', label='analy.')
#axarr[0].plot(xx, dataOutAnaTrunc, 'g-.', label='analy. trunc.')
#axarr[0].plot(xx, dataOut, 'b:', label='openAbel')
#axarr[0].set_ylabel('value')
#axarr[0].legend()

#axarr[1].semilogy(xx[:-1]/sig, np.abs((dataOut[:-1]-dataOutAna[:-1])/dataOutAna[:-1]), 'r--', label='rel. err.')
#axarr[1].semilogy(xx[:-1]/sig, np.abs((dataOut[:-1]-dataOutAnaTrunc[:-1])/dataOutAnaTrunc[:-1]), 'b:', label='rel. err. trunc.')
#axarr[1].set_ylabel('relative error')
#axarr[1].set_xlabel('y')
#axarr[1].legend()

#mpl.tight_layout()

#mpl.show()
