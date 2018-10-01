
import numpy as np
import os.path
import datetime as dt


from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
cimport scipy.linalg.cython_blas as blas


cimport openConv.mathFun as math
cimport openConv.constants as const





############################################################################################################################################
### Fast convolution based on end corrected trapezoidal rules                                                                            ###
############################################################################################################################################



# Plan convolution with trapezoidal rule
cdef int plan_conv_trap(conv_plan* pl, int order = 2) nogil except -1:

    return 0


# Execute convolution with trapezoidal rule
cdef int execute_conv_trap(conv_plan* pl, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1:

    return 0

# Destroy convolution structs
cdef int destroy_conv_trap(conv_plan* pl) nogil except -1:

    return 0

	




############################################################################################################################################









