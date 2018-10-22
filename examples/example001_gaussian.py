############################################################################################################################################
# Simple example which calculates the convolution of two Guassians.
# Results are compared with the analytical solution. Mostly default parameters are used.
############################################################################################################################################


import openConv as oc
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as mpl


# Parameters and input data
order = 3
orderM1Half = int((order-1)/2)

nData = 40
xMaxData = 8.
sigData = 1.1
stepSize = xMaxData/(nData-1)
xData = np.linspace(-orderM1Half*stepSize, orderM1Half*stepSize+xMaxData, nData+2*orderM1Half)
data = np.exp(-0.5*xData**2/sigData**2)
#data[:] = 0.
#data[orderM1Half+nData-1] = 1.

# Kernel
sigKernel = 0.6
def kern(xx):
    return np.exp(-0.5*xx**2/sigKernel**2)
nKernel = 100


# Parameters and output result
sigResult = np.sqrt(sigData**2+sigKernel**2)
nResult = nData+nKernel-1   # Can be chosen arbitrary though
xResult = np.linspace(0., (nResult-1)*stepSize, nResult)
beta = sigData*sigKernel/sigResult
alpha = sigKernel**2*xResult/sigResult**2

def phi(xx):
    return 0.5*(1.+erf(xx/np.sqrt(2.)))
    
resultAna = np.sqrt(2.*np.pi)*beta*np.exp(-0.5*xResult**2/sigResult**2) * \
            (phi((xResult+xMaxData-alpha)/beta)-phi((xResult-xMaxData-alpha)/beta))

xKernel = np.linspace(0., (nResult+nData-2)*stepSize, nResult+nData-1)
kernel = kern(xKernel)

############################################################################################################################################
# Create convolution object, which does all precomputation possible without knowing the exact 
# data. This way it's much faster if repeated convolutions with the same kernel are done.
convObj = oc.Conv(nData, 2, kern, None, 2, stepSize, nResult, method = 0, order = order)    
result = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

convObj = oc.Conv(nData, 2, kern, None, 2, stepSize, nResult, method = 1, order = order)    
result2 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

#convObj = oc.Conv(nData, 2, kern, None, 2, stepSize, nResult, method = 2, order = order)    
#result3 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

#convObj = oc.Conv(nData, 2, kern, None, 2, stepSize, nResult, method = 3, order = order)    
#result4 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

convObj = oc.Conv(nData, 2, kern, None, 2, stepSize, nResult, method = 4, order = order)    
result5 = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

mpl.figure()
#mpl.plot(xData, data, label='data')
#mpl.plot(xKernel, kernel, label='kernel')
mpl.plot(xResult, result, label='trap')
mpl.plot(xResult, result2, label='fft')
#mpl.plot(xResult, result3, 'r:', label='fmmCheb')
#mpl.plot(xResult, result4, 'g--', label='fmmExpCheb')
#mpl.plot(xResult, resultAna)
mpl.legend()

mpl.figure()
#mpl.semilogy(xData, data, label='data')
#mpl.semilogy(xKernel, kernel, label='kernel')
mpl.semilogy(xResult, result, label='trap')
mpl.semilogy(xResult, result2, label='fft')
mpl.semilogy(xResult, result5, label='fftExp')
#mpl.semilogy(xResult, result3, label='fmmCheb')
#mpl.semilogy(xResult, result4, label='fmmExpCheb')
#mpl.semilogy(xResult, resultAna)
mpl.legend()

#mpl.figure()
##mpl.semilogy(xResult, np.abs(result-result))
##mpl.semilogy(xResult, np.abs(result2-result), label='fft')
#mpl.semilogy(xResult, np.abs(result3-result), label='fmmCheb')
#mpl.semilogy(xResult, np.abs(result4-result), label='fmmExpCheb')
#mpl.legend()

mpl.figure()
#mpl.semilogy(xResult[:-1], np.clip(np.abs((result3[:-1]-result[:-1])/result[:-1]),1.e-17,np.inf), label='fmmCheb')
#mpl.semilogy(xResult[:-1], np.clip(np.abs((result4[:-1]-result[:-1])/result[:-1]),1.e-17,np.inf), label='fmmExpCheb')
mpl.semilogy(xResult[:-1], np.clip(np.abs((result[:-1]-resultAna[:-1])/resultAna[:-1]),1.e-17,np.inf), label='trap')
mpl.semilogy(xResult[:-1], np.clip(np.abs((result2[:-1]-resultAna[:-1])/resultAna[:-1]),1.e-17,np.inf), label='fft')
mpl.semilogy(xResult[:-1], np.clip(np.abs((result5[:-1]-resultAna[:-1])/resultAna[:-1]),1.e-17,np.inf), label='fftExp')
#mpl.semilogy(xResult[:-1], np.abs((result3[:-1]-resultAna[:-1])/resultAna[:-1]))
#mpl.semilogy(xResult[:-1], np.abs((result4[:-1]-resultAna[:-1])/resultAna[:-1]))
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
