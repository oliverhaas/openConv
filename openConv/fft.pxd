

from openConv.base cimport conv_plan


cdef int plan_conv_fft(conv_plan* plan) nogil except -1
cdef int execute_conv_fft(conv_plan* plan, double* dataIn, double* dataOut) nogil except -1
cdef int destroy_conv_fft(conv_plan* plan) nogil except -1

#cdef int plan_conv_fftExp(conv_plan* plan, double eps = ?) nogil except -1
#cdef int execute_conv_fftExp(conv_plan* plan, double* dataIn, double* dataOut) nogil except -1
#cdef int destroy_conv_fftExp(conv_plan* plan) nogil except -1
