
from libc.stdlib cimport malloc, calloc, free

import numpy as np

cimport openConv.constants as const
cimport openConv.mathFun as math

# Import of fft
cdef extern from 'fftw3.h':

    # Memory aligned malloc and free
    void* fftw_malloc(size_t n) nogil
    void fftw_free(void *p) nogil
    
    # Double precision plans
    ctypedef struct fftw_plan_struct:
        pass
    ctypedef fftw_plan_struct* fftw_plan

    ctypedef unsigned int fftw_r2r_kind
    # Real-to-real transforms
    fftw_plan fftw_plan_r2r_1d(int n, double* inData, double* outData, fftw_r2r_kind kind, unsigned flags) nogil
    
    # Execute plan
    void fftw_execute(const fftw_plan plan) nogil

    # Destroy plan
    void fftw_destroy_plan(fftw_plan plan) nogil

# Direction enum standard fft
cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1

# Documented flags
cdef enum:
    FFTW_ESTIMATE = 64

# Transform type real to real fft
cdef enum:
     FFTW_REDFT00 = 3
     FFTW_REDFT01 = 4
     FFTW_REDFT10 = 5
     FFTW_REDFT11 = 6



############################################################################################################################################



cdef int chebRoots(int order, double* roots) nogil:
    
    cdef:
        int ii
    
    for ii in range(order):
        roots[ii] = math.cos(0.5*const.pi*(2.*ii+1.)/order)
    
    return 0



cdef double lagrangePolInt(double xx, int ind, double* nodes, int order) nogil:

    cdef:
        int ii
        double res
    
    res = 1.
    for ii in range(ind):
        res *= (xx - nodes[ii])/(nodes[ind] - nodes[ii])
    for ii in range(ind+1, order):
        res *= (xx - nodes[ii])/(nodes[ind] - nodes[ii])

    return res



cdef int chebWeights(int order, double* weights) nogil:
    
    cdef:
        int ii
    
    for ii in range(order):
        weights[ii] = (-1.)**ii*math.sin(0.5*const.pi*(2.*ii+1.)/order)
    
    return 0



# Barycentric interpolation
cdef double barycentricInt(double xx, double* funvals, double* nodes, double* weights, int order) nogil:

    cdef:
        int ii
        double num, denom, temp
    
    num = 0.
    denom = 0.
    for ii in range(order):
        temp = weights[ii]/(xx - nodes[ii])
        num += funvals[ii]*temp
        denom += temp
        # Should make it overflow safe and zero divison safe
        if (math.fabs(num) == math.INFINITY or math.fabs(denom) == math.INFINITY): 
            return funvals[ii]
    
    return num/denom



cdef int estimateOrderCheb(funPtr fun, void* funPar, double aa, double bb, double eps, int nn, int nMax = 1000) nogil except -1:

    cdef:
        double* chebRts
        double* funVals
        double* chebCoeffs
        int ii, nFound, indMax
        fftw_plan pl
        double xx, chebCoeffsMax
        double epsDCT = const.machineEpsilon*1.e3   # DCT noise level seems to be around that range, maybe nn*machEps or nn**2*machEps???
    
            
    nn = max(nn,5) # Hard lower limit for all applications due to adaptive truncation criteria

    eps = max(epsDCT, eps)   
    chebRts = <double*> malloc(nn*sizeof(double))
    chebRoots(nn, chebRts)
    
    funVals = <double*> fftw_malloc(nn*sizeof(double))
    chebCoeffs = <double*> fftw_malloc(nn*sizeof(double))
    pl = fftw_plan_r2r_1d(nn, funVals, chebCoeffs, FFTW_REDFT10, FFTW_ESTIMATE)  # Make fft plan
    for ii in range(nn):
        xx = 0.5*(1.+chebRts[ii])*(bb-aa) + aa
        fun(&xx, funPar, &funVals[ii])
    
#    # TESTING
#    with gil:
#        xnp = np.zeros(nn)
#        valnp = np.zeros(nn)
#        for ii in range(nn):
#            xnp[ii] = chebRts[ii]
#            valnp[ii] = funVals[ii]
#        mpl.plot(xnp,valnp)
#        mpl.show()
        
    free(chebRts)   # Don't need anymore, free asap
    
    fftw_execute(pl)    # Execute fft plan
    fftw_destroy_plan(pl)   # Destroy fft plan

    free(funVals)  # Don't need anymore, free asap
    
    # Scale and take absolute value of all coefficients
    chebCoeffsMax = 0.
    for ii in range(nn):
        chebCoeffs[ii] = math.fabs(chebCoeffs[ii])
        if chebCoeffs[ii] >= chebCoeffsMax:
            chebCoeffsMax = chebCoeffs[ii]
            indMax = ii
    
#    # TESTING
#    with gil:
#        for ii in range(nn):
#            print ii, chebCoeffs[ii], nn, nMax
    if chebCoeffsMax == 0.:
        return 1
        
    for ii in range(indMax+1,nn):
        chebCoeffs[ii] /= chebCoeffsMax
    
    # Find required N (or not)
    nFound = -1
    for ii in range(indMax+3,nn):
        if (chebCoeffs[ii] + chebCoeffs[ii-1] + chebCoeffs[ii-2] <= eps):
            nFound = ii-2
            break
    if nFound > 0:  # Success
        fftw_free(chebCoeffs)
        return nFound
    
    # Not found? try again for larger number
    chebCoeffsMax = max(chebCoeffs[nn-1], chebCoeffs[nn-2], chebCoeffs[nn-3])
    nFound = min(<int> (1.1*math.log(eps)/math.log(chebCoeffsMax)*(nn-indMax)), 2*nn)   # Assume exponential decrease & safety factor
    fftw_free(chebCoeffs)
    
    if nFound >= nMax: # Hard upper limit
#        with gil:
#            print 'WARNING: Chebyshev order estimation did not converge, but reached maximum order.'
        return nMax
        
    return estimateOrderCheb(fun, funPar, aa, bb, eps, nFound, nMax = nMax)











