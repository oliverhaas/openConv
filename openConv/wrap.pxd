


cimport openConv.base as base

ctypedef int (*funPtr)(double*, void*, double*) nogil


cdef class Conv(object):
    
    cdef:
        base.conv_plan* plan
        object kernelFunPy
