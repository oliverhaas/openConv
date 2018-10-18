

from openConv.base cimport conv_plan
ctypedef int (*funPtr)(double*, void*, double*) nogil   # Importing does not work for some reason, so redefine here


cdef int plan_conv_fmmCheb(conv_plan* plan, funPtr kernelFun = ?, void* kernelFunPar = ?, double eps = ?) nogil except -1
cdef int execute_conv_fmmCheb(conv_plan* plan, double* dataIn, double* dataOut) nogil except -1
cdef int destroy_conv_fmmCheb(conv_plan* plan) nogil except -1

cdef int plan_conv_fmmExpCheb(conv_plan* plan, funPtr kernelFun = ?, void* kernelFunPar = ?, double eps = ?) nogil except -1
cdef int execute_conv_fmmExpCheb(conv_plan* plan, double* dataIn, double* dataOut) nogil except -1
cdef int destroy_conv_fmmExpCheb(conv_plan* plan) nogil except -1
