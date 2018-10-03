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
orderM1Half = int((order-1)/2)

nData = 2000
xMaxData = 8.
sigData = 1.
stepSize = xMaxData/(nData-1)
xData = np.linspace(-orderM1Half*stepSize, orderM1Half*stepSize+xMaxData, nData+2*orderM1Half)
data = np.exp(-0.5*xData**2/sigData**2)

# Kernel
sigKernel = 1.5
def kernGauss(xx):
    return np.exp(-0.5*xx**2/sigKernel**2)
nKernel = 1000

#print stepSize

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
kernel = kernGauss(xKernel)

############################################################################################################################################
# Create convolution object, which does all precomputation possible without knowing the exact 
# data. This way it's much faster if repeated convolutions with the same kernel are done.
convObj = oc.Conv(nData, 2, kernGauss, None, 2, stepSize, nResult, method = 1, order = order)    
                 
result = convObj.execute(data, leftBoundary = 3, rightBoundary = 3)

#print 'example'
#
#print result[0]
#test = (kernel[0]*data[0] + 2*kernel[1]*data[1] + 2*kernel[2]*data[2])*stepSize
#print test
#print test-result[0]
#for ii in range(3):
#    print ii, data[ii]
#for ii in range(3):
#    print ii, kernel[ii]

    
mpl.plot(xData, data)
mpl.plot(xKernel, kernel)
mpl.plot(xResult, result)
mpl.plot(xResult, resultAna)

mpl.figure()
mpl.semilogy(xData, data)
mpl.semilogy(xKernel, kernel)
mpl.semilogy(xResult, result)
mpl.semilogy(xResult, resultAna)

mpl.figure()
mpl.semilogy(xResult, np.abs(result-resultAna))

mpl.figure()
mpl.semilogy(xResult[:-1], np.abs((result[:-1]-resultAna[:-1])/resultAna[:-1]))
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
