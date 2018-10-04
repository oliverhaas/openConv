
import numpy as np
import datetime as dt


from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
cimport scipy.linalg.cython_blas as blas


cimport openConv.mathFun as math
cimport openConv.constants as const
cimport openConv.base as base




############################################################################################################################################
### Convolution based on end corrected trapezoidal/midpoint rules                                                                        ###
############################################################################################################################################

ctypedef struct methodData_trap:
    int nothingHere


# Plan convolution with trapezoidal rule
cdef int plan_conv_trap(conv_plan* pl) nogil except -1:

    cdef:
        int ii, jj
        methodData_trap* md

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
        int orderM1Half = <int> ((pl.order-1)/2)

    # Input check
    if NULL == pl or NULL == dataIn or NULL == dataOut:
        with gil:
            raise ValueError('Illegal input argument; something is NULL.')   
        
    # Symmetries
    signData = base.symSignFac(pl.symData)
    signKernel = base.symSignFac(pl.symKernel)

    # Normal convolution
    for ii in range(pl.nData):
        # kernel right
        for jj in range(pl.nData-ii):
            dataOut[ii] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii+jj]
    for ii in range(1,pl.nDataOut):
        # kernel left to axis right
        for jj in range(max(1,ii-pl.nData+1),ii+1):
            dataOut[ii] += pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii-jj]
    for ii in range(pl.nDataOut):
        # kernel left after axis left
        for jj in range(ii+1,pl.nData+ii):
            dataOut[ii] += signData*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+jj-ii]

    return 0


# Destroy convolution structs
cdef int destroy_conv_trap(conv_plan* pl) nogil except -1:

    cdef:
        methodData_trap* md = <methodData_trap*> pl.methodData

    free(md)
    
    return 0

	




############################################################################################################################################









