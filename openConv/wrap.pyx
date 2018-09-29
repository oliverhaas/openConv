

import sys
import numpy

cimport base



cdef class Conv(object):


    def __init__(self, int nData, double shift, double stepSize, 
                 int method = 3, int order = 2, double eps = 1.e-15):
        
        try:
            self.plan = base.plan_conv(nData, shift, stepSize, 
                                       method = method, order = order, eps = eps)
        except:
            print "Unexpected error in Cython routines:", sys.exc_info()[0], sys.exc_info()[1]
            raise


    def execute(self, double[:] dataIn, int leftBoundary = 0, int rightBoundary = 0):

        cdef:
            double[::1] dataInTemp
            double[::1] dataOut

        dataInTemp = numpy.copy(dataIn)
        dataOut = numpy.copy(dataInTemp)

        try:
            base.execute_conv(self.plan, &dataInTemp[0], &dataOut[0], leftBoundary = leftBoundary, rightBoundary = rightBoundary)
        except:
            print "Unexpected error in Cython routines:", sys.exc_info()[0], sys.exc_info()[1]
            raise        

        return numpy.asarray(dataOut)[:self.plan.nData]


    def __dealloc__(self):

        base.destroy_conv(self.plan)

