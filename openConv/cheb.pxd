
ctypedef int (*funPtr)(double*, void*, double*) nogil   # Importing does not work for some reason, so redefine here

cdef int chebRoots(int order, double* roots) nogil
cdef int chebWeights(int order, double* weights) nogil
cdef int estimateOrderCheb(funPtr fun, void* funPar, double aa, double bb, double eps, int nn, int nMax = ?) nogil except -1


cdef double lagrangePolInt(double xx, int ind, double* nodes, int order) nogil
cdef double barycentricInt(double xx, double* funvals, double* nodes, double* weights, int order) nogil
