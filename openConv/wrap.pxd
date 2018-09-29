


cimport openAbel.abel.base as base


cdef class Abel(object):
    
    cdef:
        base.abel_plan* plan
