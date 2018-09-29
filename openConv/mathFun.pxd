cdef extern from "math.h":
    double log(double x) nogil
    double log10(double x) nogil
    double log2(double x) nogil
    double sqrt(double x) nogil
    double sin(double x) nogil
    double sinh(double x) nogil
    double tanh(double x) nogil
    double cos(double x) nogil
    double tan(double x) nogil
    double acos(double x) nogil
    double atan(double x) nogil
    double pow(double x, double y) nogil
    double exp(double x) nogil
    double fabs(double x) nogil
    double erf(double x) nogil
    double erfc(double x) nogil
    double lgamma(double x) nogil
    double tgamma(double x) nogil
    double round(double x) nogil
    double fmin(double x, double y) nogil
    double fmax(double x, double y) nogil
    int isnan(double x) nogil
    double INFINITY

cdef extern from "complex.h":
    double creal(double complex z) nogil
    double cimag(double complex z) nogil
    double cabs(double complex z) nogil
    double complex cexp(double complex z) nogil
    double carg(double complex z) nogil
    double complex csqrt(double complex z) nogil
    double complex catan(double complex z) nogil



cdef unsigned int uintMax(unsigned int aa, unsigned int bb) nogil
cdef unsigned int uintMin(unsigned int aa, unsigned int bb) nogil
