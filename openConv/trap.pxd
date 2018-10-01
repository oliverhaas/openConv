

from openConv.base cimport conv_plan



cdef int plan_conv_trap(conv_plan* plan, int order = ?) nogil except -1
cdef int execute_conv_trap(conv_plan* plan, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1
cdef int destroy_conv_trap(conv_plan* plan) nogil except -1
