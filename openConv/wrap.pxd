


cimport openConv.base as base


cdef class Conv(object):
    
    cdef:
        base.conv_plan* plan
