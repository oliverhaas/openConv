



ctypedef int (*interpFun)(double* xx, void* params, double* out) nogil

ctypedef struct Interpolator:
    void* helper
    int (*interpolate)(Interpolator* me, double* xx, double* out) nogil
    int (*interpolateD)(Interpolator* me, double* xx, double* out, int deriv) nogil
    void (*free)(Interpolator* me) nogil


cdef double polInt(double* data, int nData, double xx) nogil

cdef Interpolator* Spline1D1DEquiFromData(double* yy, double xMin, double xMax, int nx, int degree = ?, int nExtLeft = ?,
                                          int typeLeft = ?, int nDersLeft = ?, int* derOrdsLeft = ?, double* derValsLeft = ?, 
                                          int nExtRight = ?, int typeRight = ?, int nDersRight = ?, int* derOrdsRight = ?, 
                                          double* derValsRight = ?) nogil except NULL
cdef double bSplineEqui(double xx, int degree, int ex = ?, int exInterv = ?) nogil
cdef double bSplineEquiD(double xx, int degree, int deriv, int ex = ?, int exInterv = ?) nogil
cdef double bStepSplineEqui(double xx, int degree) nogil

cdef Interpolator* Newton1D1DEquiFromData(double* yy, double xMin, double xMax, int nx, int degree = ?) nogil except NULL

