
import numpy as np
import os.path
import datetime as dt


from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
cimport scipy.linalg.cython_blas as blas


cimport openConv.mathFun as math
cimport openConv.constants as const
cimport openConv.coeffs as coeffs
cimport openConv.base as base




############################################################################################################################################
### Fast convolution based on end corrected trapezoidal rules                                                                            ###
############################################################################################################################################

ctypedef struct methodData_trap:
    int nothingHere


# Plan convolution with trapezoidal rule
cdef int plan_conv_trap(conv_plan* pl) nogil except -1:

    cdef:
        int ii, jj
        methodData_trap* md
        double[::1] coeffs_smooth_memView

    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument; plan is NULL.')   

    # Main method struct
    md = <methodData_trap*> malloc(sizeof(methodData_trap))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    pl.methodData = <void*> md
       
    return 0


# Execute convolution with trapezoidal rule
cdef int execute_conv_trap(conv_plan* pl, double* dataIn, double* dataOut) nogil except -1:

    cdef:
        int ii, jj
        double temp, signData, signKernel
        int orderM1Half, orderM1HalfInner, orderM1HalfDiff, orderM1HalfSum
        int nDataOut = pl.nData+pl.nKernel-1
        
    # Symmetries
    signData = base.symSignFac(pl.symData)
    signKernel = base.symSignFac(pl.symKernel)

    orderM1Half = <int> ((pl.order-1)/2)
    orderM1HalfInner = <int> (pl.order/2)
    orderM1HalfDiff = orderM1HalfInner - orderM1Half
    orderM1HalfSum = orderM1HalfInner + orderM1Half
            
    # Set output to zero first
    memset(dataOut, 0, nDataOut*sizeof(double))

    # Normal convolution
    for ii in range(pl.nData-1):
        # kernel right
        for jj in range(pl.nData-ii):
            dataOut[ii] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii+jj]
    for ii in range(1,nDataOut):
        # kernel left to axis right
        for jj in range(max(0,ii-pl.nData+1),ii+1):
            dataOut[ii] += pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii-jj]
    for ii in range(nDataOut):
        # kernel left after axis left
        for jj in range(ii,pl.nData+ii):
            dataOut[ii] += signData*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+jj-ii]

    # End corrections
    # TODO check maybe, and make function in base.pyx ???
    for ii in range(pl.nData-1):
        for jj in range(pl.order): # Central kernel right
            dataOut[ii] += signKernel*pl.coeffsSmooth[pl.order-1-jj]*pl.kernel[jj]*dataIn[ii+jj]
        for jj in range(pl.order): # Tail right
            dataOut[ii] += signKernel*pl.coeffsSmooth[jj]*pl.kernel[pl.nData-ii-1+jj-orderM1HalfDiff]*dataIn[pl.nData-1+jj-orderM1HalfDiff]
    for ii in range(1,pl.nData):
        for jj in range(pl.order): # Central kernel left
            dataOut[ii] += pl.coeffsSmooth[jj]*pl.kernel[pl.order-1-jj]*dataIn[ii+jj]
    for ii in range(1,nDataOut):
        for jj in range(pl.order): # Axis right
            dataOut[ii] += pl.coeffsSmooth[pl.order-1-jj]*pl.kernel[ii+pl.order-1-jj]*dataIn[jj]
    for ii in range(nDataOut):
        for jj in range(pl.order): # Axis left
            dataOut[ii] += signData*pl.coeffsSmooth[jj]*pl.kernel[ii+pl.order-1-jj]*dataIn[pl.order-1-jj]
    for ii in range(nDataOut):
        for jj in range(pl.order):    # Tail left axis left
            dataOut[ii] += signData*pl.coeffsSmooth[pl.order-1-jj] * \
                           pl.kernel[pl.nData+ii-2+pl.order-orderM1HalfDiff-jj]*dataIn[pl.nData-2+pl.order-orderM1HalfDiff-jj]

    for ii in range(nDataOut):
        dataOut[ii] *= pl.stepSize

    if signData*signKernel < 0.:
        dataOut[0] = 0.

    return 0


# Destroy convolution structs
cdef int destroy_conv_trap(conv_plan* pl) nogil except -1:

    cdef:
        methodData_trap* md = <methodData_trap*> pl.methodData

    free(md)
    
    return 0

	




############################################################################################################################################









