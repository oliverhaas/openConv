

from openAbel.abel.base cimport abel_plan



cdef int plan_fat_trapezoidalDesingConst(abel_plan* plan) nogil except -1
cdef int execute_fat_trapezoidalDesingConst(abel_plan* plan, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1
cdef int destroy_fat_trapezoidalDesingConst(abel_plan* plan) nogil except -1

cdef int plan_fat_trapezoidalEndCorr(abel_plan* plan, int order = ?) nogil except -1
cdef int execute_fat_trapezoidalEndCorr(abel_plan* plan, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1
cdef int destroy_fat_trapezoidalEndCorr(abel_plan* plan) nogil except -1
