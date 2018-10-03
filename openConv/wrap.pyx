

import sys
import numpy as np

cimport base
cimport openConv.interpolate as interp


cdef class Conv(object):

 
    def __init__(self, int nData, int symData, object kernelFunPy, double[:] kernelIn, int symKernel, double stepSize, int nDataOut,
                 double shiftData = 0., double shiftKernel = 0., int leftBoundaryKernel = 0, int rightBoundaryKernel = 0,
                 int method = 0, int order = 1, double eps = 1.e-15):
      
        cdef:
            double[::1] xTemp
            double xMin, xMax
            double[::1] kernel
            int nKernel, orderM1Half
        
        if not (symKernel == 1 or symKernel == 2) or not (symData == 1 or symData == 2) or \
           not (shiftData == 0.) or not (shiftKernel == 0.):
            raise NotImplementedError('Not implemented for given parameters.')

        orderM1Half = <int> ((order-1)/2)
            
        if kernelFunPy == None and kernelIn == None:
            raise TypeError('At least one of kernel function or kernel vector has to be not None.')
        elif kernelIn == None:
            nKernel = nDataOut+nData-1      # Is different for symData = 0 (@future)
            xMin = (-orderM1Half+shiftData)*stepSize
            xMax = (orderM1Half+shiftData+nKernel-1)*stepSize
            xTemp = np.linspace(xMin, xMax, nKernel+2*orderM1Half)
            kernel = kernelFunPy(np.asarray(xTemp))
            leftBoundaryKernel = 3
            rightBoundaryKernel = 3
        else:
            nKernel = kernelIn.shape[0] - ((leftBoundaryKernel == 3) + (rightBoundaryKernel == 3))*orderM1Half
            kernel = np.ascontiguousarray(kernelIn)     # Ensure C contiguous data
        try:
            self.plan = base.plan_conv(nData, symData, &kernel[0], nKernel, symKernel, stepSize, nDataOut,
                                       kernelFun = NULL, kernelFunPar = NULL, shiftData = shiftData, shiftKernel = shiftKernel,
                                       leftBoundaryKernel = leftBoundaryKernel, rightBoundaryKernel = rightBoundaryKernel, method = method, 
                                       order = order, eps = eps)
        except:
            print "Unexpected error in Cython routines:", sys.exc_info()[0], sys.exc_info()[1]
            raise


    def execute(self, double[:] dataIn, int leftBoundary = 0, int rightBoundary = 0):

        cdef:
            double[::1] dataInTemp
            double[::1] dataOut

        dataInTemp = np.ascontiguousarray(dataIn)   # Ensure C contiguous data
        dataOut = np.empty(self.plan.nDataOut)

        try:
            base.execute_conv(self.plan, &dataInTemp[0], &dataOut[0], leftBoundary = leftBoundary, rightBoundary = rightBoundary)
        except:
            print "Unexpected error in Cython routines:", sys.exc_info()[0], sys.exc_info()[1]
            raise        

        return np.asarray(dataOut)


    def __dealloc__(self):
        base.destroy_conv(self.plan)

