

import datetime
import matplotlib.pyplot as mpl
import numpy

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

cimport scipy.linalg.cython_blas as blas

cimport openConv.mathFun as math
cimport openConv.constants as const
cimport openConv.base as base
cimport openConv.interpolate as interp
cimport openConv.cheb as cheb

# Calling blas functions needs some simple constants
cdef:
    int ONE = 1
    int MONE = -1
    double ZEROD = 0.
    double ONED = 1.

############################################################################################################################################
### Fast multipole method trapezoidal convolution                                                                                        ###
############################################################################################################################################

ctypedef struct methodData_fmmCheb:
    double* chebRoots
    int* kl
    int* klCum
    double* mtmp
    double* mtmm
    double* ltp
    double* mtl
    int pp
    int pp1
    int ss    
    int nlevs
    int kTotal
    interp.Interpolator* kernelInterp
    funPtr kernelFun
    void* kernelFunPar

# Plan FMM
cdef int plan_conv_fmmCheb(conv_plan* pl, funPtr kernelFun = NULL, void* kernelFunPar = NULL, double eps = 1.e-15) nogil except -1:

    cdef:
        methodData_fmmCheb* md
        double temp0
        int orderM1Half
        int ii, jj, ll
        
    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument; plan is NULL.')   
    if pl.nData < 4:
        with gil:
            raise ValueError('Not enough data points for method.')   
            
    # Main method struct
    md = <methodData_fmmCheb*> malloc(sizeof(methodData_fmmCheb))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.chebRoots = NULL
        md.kl = NULL
        md.klCum = NULL
        md.mtmp = NULL
        md.mtmm = NULL
        md.ltp = NULL
        md.mtl = NULL
        md.kernelInterp = NULL
        md.kernelFun = NULL
        md.kernelFunPar = NULL
    pl.methodData = <void*> md

    # Not implemented stuff catch here
    if (pl.symData != 1 and pl.symData != 2) or (pl.symKernel != 1 and pl.symKernel != 2) or (pl.shiftKernel != 0. or pl.shiftData != 0.):
        destroy_conv_fmmCheb(pl)
        with gil:
            raise NotImplementedError('Not implemented (yet).')

    # Helper stuff
    orderM1Half = <int> ((pl.order-1)/2)
    
    # Handling of kernel function
    if NULL == kernelFun:
        md.kernelInterp = interp.Newton1D1DEquiFromData(pl.kernel, (pl.shiftKernel-orderM1Half)*pl.stepSize, 
                                                        (pl.nKernel-1+pl.shiftKernel+orderM1Half)*pl.stepSize, pl.nKernel+2*orderM1Half, 
                                                        degree = max(pl.order-1,1))
        md.kernelFun = &kernelFunHelper
        md.kernelFunPar = <void*> md.kernelInterp
    else:
        md.kernelFun = kernelFun
        md.kernelFunPar = kernelFunPar
    
    # Hierarchical decomposition
    # Calculate needed order for desired precision
    md.nlevs = max(<int> ( math.log2((pl.nDataOut-1.)/6.) + 1. ), 2) # Rough estimate to check
    md.pp1 = 0
    temp0 = 2.*pl.stepSize*(pl.nKernel+pl.shiftKernel-1)
    for ii in range(md.nlevs):
        temp0 *= 0.5
        md.pp1 = max(md.pp1, cheb.estimateOrderCheb(md.kernelFun, md.kernelFunPar, temp0*0.25, temp0*0.75, eps, 20, nMax = 100))
        md.pp1 = max(md.pp1, cheb.estimateOrderCheb(md.kernelFun, md.kernelFunPar, temp0*0.5, temp0, eps, 20, nMax = 100))
    md.pp = max(2, md.pp1-1)
    md.pp1 = md.pp + 1
    md.nlevs = max(<int> ( math.log2((pl.nDataOut-1.)/(2.*md.pp)) + 1. ), 2)
    md.ss = max(<int> ( (pl.nDataOut-1.)/2**md.nlevs + 1. ), 3)                              # ss ~= 2*pp theoretical
    md.kTotal = 2**(md.nlevs+1) - 1                                                       # Total number of intervals in all levels

    # Allocation of arrays part
    md.chebRoots = <double*> malloc(md.pp1*sizeof(double))
    md.kl = <int*> malloc((md.nlevs+1)*sizeof(int))
    md.klCum = <int*> malloc((md.nlevs+1)*sizeof(int))
    md.mtmp = <double*> malloc(md.pp1**2*sizeof(double))
    md.mtmm = <double*> malloc(md.pp1**2*sizeof(double))
    md.ltp = <double*> malloc(md.pp1*md.ss*sizeof(double))
    md.mtl = <double*> calloc(2*md.nlevs*md.pp1**2, sizeof(double))
    
    if (NULL == md.chebRoots or NULL == md.kl or NULL == md.klCum or NULL == md.mtmp or 
        NULL == md.mtmm or NULL == md.ltp or NULL == md.mtl):
        destroy_conv_fmmCheb(pl)
        with gil:        
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

    # More hierarchical decomposition stuff
    md.klCum[0] = 0
    md.kl[0] = 2**md.nlevs
    for ii in range(1,md.nlevs+1):
        md.kl[ii] = 2**(md.nlevs-ii)
        md.klCum[ii] = md.klCum[ii-1] + md.kl[ii-1]
    cheb.chebRoots(md.pp1, md.chebRoots)

    # Moment to moment coefficients
    for ii in range(md.pp1):
        for jj in range(md.pp1):
            md.mtmp[ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*md.chebRoots[jj]+0.5, ii, md.chebRoots, md.pp1)
            md.mtmm[ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*md.chebRoots[jj]-0.5, ii, md.chebRoots, md.pp1)    

    # Local to potential coefficients
    for ii in range(md.ss):
        temp0 = 2.*(ii+1)/md.ss-1.
        for jj in range(md.pp1):
            md.ltp[ii*md.pp1+jj] = cheb.lagrangePolInt(temp0, jj, md.chebRoots, md.pp1)

    # Moment to local coefficients
    for ll in range(md.nlevs):
        for ii in range(md.pp1):
            for jj in range(md.pp1):
                temp0 = md.kl[md.nlevs-ll]*(2.+0.5*(md.chebRoots[jj]-md.chebRoots[ii]))*md.ss*pl.stepSize
                md.kernelFun(&temp0, md.kernelFunPar, &md.mtl[2*ll*md.pp1**2+ii*md.pp1+jj])
                temp0 = md.kl[md.nlevs-ll]*(3.+0.5*(md.chebRoots[jj]-md.chebRoots[ii]))*md.ss*pl.stepSize
                md.kernelFun(&temp0, md.kernelFunPar, &md.mtl[(2*ll+1)*md.pp1**2+ii*md.pp1+jj])

    return 0


# Execute FMM
cdef int execute_conv_fmmCheb(conv_plan* pl, double* dataIn, double* dataOut) nogil except -1:

    cdef:
        int ii, jj, kk, ll, mm, nn
        double* moments
        double* local
        int orderM1Half
        double temp0, signData, signKernel
        methodData_fmmCheb* md
    
    # Helper stuff
    orderM1Half = <int> ((pl.order-1)/2)
    md = <methodData_fmmCheb*> pl.methodData
    
    # Symmetries
    signData = base.symSignFac(pl.symData)
    signKernel = base.symSignFac(pl.symKernel)

    # Memory allocation
    moments = <double*> calloc(md.kTotal*md.pp1, sizeof(double))
    local = <double*> calloc(md.kTotal*md.pp1, sizeof(double))
    if (NULL == moments or NULL == local):
        free(moments)
        free(local)
        with gil:        
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

    # Finest level moment calculation   # TODO
    mm = <int> (pl.nData-2)/md.ss
    blas.dgemm('n', 'n', &md.pp1, &mm, &md.ss, &ONED, md.ltp, &md.pp1, 
               &dataIn[orderM1Half+1], &md.ss, &ZEROD, moments, &md.pp1)
    for ii in range(mm*md.ss+1, pl.nData):
        nn = (ii-1) - mm*md.ss
        for jj in range(md.pp1):
            moments[mm*md.pp1+jj] += md.ltp[nn*md.pp1+jj]*dataIn[orderM1Half+ii]

    # Upward Pass / Moment to moment
    mm = 2*md.pp1
    for ll in range(1, md.nlevs):
        blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmm, &md.pp1, 
                   &moments[md.klCum[ll-1]*md.pp1], &mm, &ZEROD, &moments[md.klCum[ll]*md.pp1], &md.pp1)
        blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmp, &md.pp1, 
                   &moments[(md.klCum[ll-1]+1)*md.pp1], &mm, &ONED, &moments[md.klCum[ll]*md.pp1], &md.pp1)

    # Interaction Phase / Moment to local
    for ll in range(md.nlevs):
    
        # Right kernel
        for kk in range(0, md.kl[ll]-2, 2): # If even
            blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
            blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+3)*md.pp1], &ONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
        
        for kk in range(1, md.kl[ll]-2, 2): # If odd     
            blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
                       
        # Left kernel right axis
        for kk in range(3, md.kl[ll], 2):   # If odd
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]-2)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]-3)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
        for kk in range(2, md.kl[ll], 2):   # If even    
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]-2)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
                       
        # Left kernel left axis
        # If even (kk = 0)
        blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                   &moments[(md.klCum[ll]+1)*md.pp1], &MONE, &ONED, &local[md.klCum[ll]*md.pp1], &ONE)
        # If odd (kk = 1)
        blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                   &moments[md.klCum[ll]*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+1)*md.pp1], &ONE)
        blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                   &moments[(md.klCum[ll]+1)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+1)*md.pp1], &ONE) 

    # Downward Pass / Local to local
    mm = 2*md.pp1
    for ll in range(md.nlevs-1, 0, -1):
        blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmm, &md.pp1, 
                   &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[md.klCum[ll-1]*md.pp1], &mm)
        blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmp, &md.pp1, 
                   &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[(md.klCum[ll-1]+1)*md.pp1], &mm)

    # Potential evaluation / local to potential
    mm = (pl.nDataOut-2)/md.ss
    blas.dgemm('t', 'n', &md.ss, &mm, &md.pp1, &ONED, md.ltp, &md.pp1, local, &md.pp1, &ONED, &dataOut[1], &md.ss)
    for ii in range(mm*md.ss+1, pl.nDataOut):
        nn = (ii-1) - mm*md.ss
        for jj in range(md.pp1):
            dataOut[ii] += md.ltp[nn*md.pp1+jj]*local[mm*md.pp1+jj]
    for jj in range(md.pp1):
        dataOut[0] += local[jj]*cheb.lagrangePolInt(-1., jj, md.chebRoots, md.pp1)

    free(moments)
    free(local)
    
    # Direct short range
    for ii in range(1,pl.nData):
        kk = (ii-1)/md.ss
        mm = min(pl.nData-ii, (kk+2)*md.ss-ii+1)
        for jj in range(mm):
            dataOut[ii] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii+jj]

    for jj in range(1, min(2*md.ss+1,pl.nData)):
        dataOut[0] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+jj]
    dataOut[0] += (signData+signKernel)*0.5*pl.kernel[orderM1Half]*dataIn[orderM1Half]
    
    for ii in range(1,pl.nDataOut):
        kk = (ii-1)/md.ss
        mm = ii-(kk-1)*md.ss
        for jj in range(max(1,1+ii-pl.nData),min(mm,ii)):
            dataOut[ii] += pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii-jj]

    for ii in range(min(md.ss+1,max(pl.nKernel-pl.nData,pl.nKernel-md.ss-1))):
         for jj in range(1,min(md.ss+1,pl.nData)):
            dataOut[ii] += signData*pl.kernel[orderM1Half+ii+jj]*dataIn[orderM1Half+jj]
    for ii in range(1,pl.nDataOut):
        dataOut[ii] += (1.+signData)*0.5*pl.kernel[orderM1Half+ii]*dataIn[orderM1Half]

    return 0


# Destroy FMM
cdef int destroy_conv_fmmCheb(conv_plan* pl) nogil except -1:

    cdef:
        methodData_fmmCheb* md
        
    if not NULL == pl:
        md = <methodData_fmmCheb*> pl.methodData
        free(md.chebRoots)
        free(md.kl)
        free(md.klCum)
        free(md.mtmp)
        free(md.mtmm)
        free(md.mtl)
        free(md.ltp)
        free(md)

    return 0

##############

cdef int kernelFunHelper(double* xx, void* par, double* out) nogil:

    cdef:
        interp.Interpolator* kernelInterp = <interp.Interpolator*> par
        
    kernelInterp.interpolate(kernelInterp, xx, out)
    
    return 0



############################################################################################################################################
### Fast multipole method trapezoidal convolution for somewhat exponentially decaying functions                                          ###
############################################################################################################################################

ctypedef struct methodData_fmmExpCheb:
    (int*) kl, klCum
    (double*) mtmp, mtmm, ltlp, ltlm, ltp, ltp0, mtl, stm, stm0
    int pp, pp1, ss, nlevs, kTotal, nlevsCut, kTotalCut, longRangeFlag


# Plan FMM
cdef int plan_conv_fmmExpCheb(conv_plan* pl, funPtr kernelFun = NULL, void* kernelFunPar = NULL, 
                              double eps = 1.e3*const.machineEpsilon) nogil except -1:

    cdef:
        methodData_fmmExpCheb* md = NULL
        double temp0, temp1, temp2
        int orderM1Half, nKernelRed, nKernelExt
        int ii, jj, kk, ll, mm, maxInd
        double* kernelLog = NULL
        kernelFunCombinerStruct kFCS

        interp.Interpolator* kI = NULL
        funPtr kF = NULL, kFL = NULL
        (void*) kFP = NULL, kFLP = NULL
        double* chebRts = NULL
        double lam, epsAbsCut = 1.e30*const.doubleMin

    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument; plan is NULL.')
    eps = max(min(eps, 1.e-3), const.machineEpsilon)
    orderM1Half = <int> ((pl.order-1)/2)
    nKernelRed = pl.nDataOut + pl.nData - 1 
    nKernelExt = 2*nKernelRed
    if pl.nData < 4:
        with gil:
            raise ValueError('Not enough data points for method.')   
            
    # Main method struct
    md = <methodData_fmmExpCheb*> malloc(sizeof(methodData_fmmExpCheb))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.kl = NULL
        md.klCum = NULL
        md.mtmp = NULL
        md.mtmm = NULL
        md.ltp = NULL
        md.ltp0 = NULL
        md.mtl = NULL
        md.stm = NULL
        md.stm0 = NULL
        md.ltlp = NULL
        md.ltlm = NULL
        md.longRangeFlag = 1
    pl.methodData = <void*> md

    # Not implemented stuff catch here
    if (pl.symData != 1 and pl.symData != 2) or (pl.symKernel != 1 and pl.symKernel != 2) or \
       (pl.shiftKernel != 0. or pl.shiftData != 0.):
        destroy_conv_fmmExpCheb(pl)
        with gil:
            raise NotImplementedError('Not implemented (yet).')

    # Check if kernel is compatible
    maxInd = 0
    for ii in range(pl.nKernel+2*orderM1Half):  # TODO better criteria?
        if pl.kernel[ii] > epsAbsCut:
            maxInd = ii+1
    maxInd = min(nKernelExt+2*orderM1Half, maxInd)
    if maxInd - 2*orderM1Half < 20:
        md.longRangeFlag = 0
        md.ss = max(maxInd,10)
    for ii in range(maxInd):    # Check if kernel is postive non-zero.
        if pl.kernel[ii] <= 0.:
            with gil:
                raise ValueError('Kernel is not positive non-zero. Maybe input negative kernel?')
    
    if md.longRangeFlag == 1:
        # Take care of log-scale kernel values and estimate rough exponential asymptotic scaling
        kernelLog = <double*> malloc((nKernelExt+2*orderM1Half)*sizeof(double))
        for ii in range(maxInd):
            kernelLog[ii] = math.log(pl.kernel[ii])
        jj = min(<int> (0.8*maxInd), maxInd-3)
        _linReg(&kernelLog[jj], maxInd-jj, &temp0, &temp1)
        lam = -temp1/pl.stepSize
        if lam < 0.:
            with gil:
                raise ValueError('Asymptotic scaling seems not to be decaying exponentially.')            
        for ii in range(maxInd, nKernelExt+2*orderM1Half):
            kernelLog[ii] = temp0 - lam*pl.stepSize*(ii-jj)
        for ii in range(jj, maxInd):
            temp1 = _smoother((ii-jj)/(maxInd-jj+1.))
            kernelLog[ii] = kernelLog[ii]*temp1 + (1.-temp1)*(temp0 - lam*pl.stepSize*(ii-jj))

        # Further handling of kernel function
        # Interpolate in log scale since function assumed to be somewhat exponential
        kI = interp.Newton1D1DEquiFromData(kernelLog, (pl.shiftKernel-orderM1Half)*pl.stepSize, 
                                           (nKernelExt-1+pl.shiftKernel+orderM1Half)*pl.stepSize, nKernelExt+2*orderM1Half, 
                                           degree = max(pl.order-1,1))
        kF = &kernelFunExpHelper
        kFP = <void*> kI
        kFL = &kernelFunHelper
        kFLP = <void*> kI

        # Calculate needed order for desired precision
        md.nlevs = max(<int> ( math.log2((pl.nDataOut-1.)/4.) + 1. ), 2) # Maximal estimate just to check
        md.ss = max(<int> ( (pl.nDataOut-1.)/2**md.nlevs + 1. ), 3)                 # ss ~= 2*pp theoretical
        md.nlevsCut = min(<int> (math.log2(1.*maxInd/md.ss) + 1.), md.nlevs)   
        md.pp1 = 3
        kFCS.kernelFunLog = kFL
        kFCS.kernelFunLogPar = kFLP
        kFCS.lam = lam
        temp1 = max(1.e3*const.machineEpsilon, eps)
        temp0 = 2.*0.5**(md.nlevs-md.nlevsCut)*pl.stepSize*(nKernelExt+pl.shiftKernel-1)
        for ii in range(md.nlevs-md.nlevsCut, md.nlevs):    # TODO check
            temp0 *= 0.5
            kFCS.delta = 0.125*temp0
            kk = min(<int> (kFCS.delta*2./pl.stepSize + 1.), 100)
            if md.pp1 > kk:
                break
            for jj in range(-1,4,1):
                kFCS.tMeanMTauMean = 0.5*temp0 + jj*kFCS.delta
                md.pp1 = max(md.pp1, cheb.estimateOrderCheb(kernelFunCombiner, &kFCS, -1., 1., temp1, min(50,kk), nMax = kk))

        free(kernelLog)
            
        # Hierarchical decomposition
        md.pp = max(2, md.pp1-1)
        md.pp1 = md.pp + 1
        md.nlevs = max(<int> ( math.log2((pl.nDataOut-1.)/(2.*md.pp)) + 1. ), 1)
        md.ss = max(<int> ( (pl.nDataOut-1.)/2**md.nlevs + 1. ), 3)                 # ss ~= 2*pp theoretical
        md.kTotal = 2**(md.nlevs+1) - 1                                             # Total number of intervals in all levels
        md.nlevsCut = min(<int> (math.log2(1.*maxInd/md.ss) + 1.), md.nlevs)   
        md.kTotalCut = md.kTotal - 2**(md.nlevs-md.nlevsCut) + 1    # TODO check nlevs definition
        
        # More allocation of other arrays part
        md.kl = <int*> malloc((md.nlevs+1)*sizeof(int))
        md.klCum = <int*> malloc((md.nlevs+1)*sizeof(int))
        md.mtmp = <double*> calloc((md.nlevsCut-1)*md.pp1**2, sizeof(double))
        md.mtmm = <double*> calloc((md.nlevsCut-1)*md.pp1**2, sizeof(double))
        md.ltp = <double*> calloc(md.pp1*md.ss, sizeof(double))
        md.ltp0 = <double*> calloc(md.pp1, sizeof(double))
        md.mtl = <double*> calloc(2*md.nlevsCut*md.pp1**2, sizeof(double))
        md.stm = <double*> calloc(md.pp1*md.ss, sizeof(double))
        md.stm0 = <double*> calloc(md.pp1, sizeof(double))
        md.ltlp = <double*> calloc((md.nlevsCut-1)*md.pp1**2, sizeof(double))
        md.ltlm = <double*> calloc((md.nlevsCut-1)*md.pp1**2, sizeof(double))
        chebRts = <double*> malloc(md.pp1*sizeof(double))
        if (NULL == md.kl or NULL == md.klCum or NULL == md.mtmp or NULL == md.mtmm or NULL == md.ltp or NULL == md.ltp0 or 
            NULL == md.mtl or NULL == chebRts or NULL == md.ltlp or NULL == md.ltlm or NULL == md.stm0):
            destroy_conv_fmmExpCheb(pl)
            with gil:        
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

        # More hierarchical decomposition stuff
        md.klCum[0] = 0
        md.kl[0] = 2**md.nlevs
        for ii in range(1,md.nlevs+1):
            md.kl[ii] = 2**(md.nlevs-ii)
            md.klCum[ii] = md.klCum[ii-1] + md.kl[ii-1]
        cheb.chebRoots(md.pp1, chebRts)

        # Moment to moment
        for ll in range(md.nlevsCut-1):
            for ii in range(md.pp1):
                for jj in range(md.pp1):
                    temp0 = math.exp( md.kl[md.nlevs-ll]*0.5*pl.stepSize*md.ss*lam )
                    temp1 = math.exp( -md.kl[md.nlevs-ll]*0.5*pl.stepSize*md.ss*lam )
                    md.mtmp[ll*md.pp1**2+ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*chebRts[jj]+0.5, ii, chebRts, md.pp1)*temp0
                    md.mtmm[ll*md.pp1**2+ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*chebRts[jj]-0.5, ii, chebRts, md.pp1)*temp1
                    
        # Local to local
        for ll in range(md.nlevsCut-1):
            for ii in range(md.pp1):
                for jj in range(md.pp1):
                    temp0 = math.exp( -md.kl[md.nlevs-ll]*0.5*pl.stepSize*md.ss*lam )
                    temp1 = math.exp( md.kl[md.nlevs-ll]*0.5*pl.stepSize*md.ss*lam )
                    md.ltlp[ll*md.pp1**2+ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*chebRts[jj]+0.5, ii, chebRts, md.pp1)*temp0
                    md.ltlm[ll*md.pp1**2+ii*md.pp1+jj] = cheb.lagrangePolInt(0.5*chebRts[jj]-0.5, ii, chebRts, md.pp1)*temp1

        # Source to moment
        for jj in range(md.pp1):
            temp1 = math.exp( -(md.ss/2.)*pl.stepSize*lam )
            md.stm0[jj] = temp1*cheb.lagrangePolInt(-1., jj, chebRts, md.pp1)
        for ii in range(md.ss):
            temp0 = 2.*(ii+1)/md.ss-1.
            for jj in range(md.pp1):
                temp1 = math.exp( (ii+1.-md.ss/2.)*pl.stepSize*lam )
                md.stm[ii*md.pp1+jj] = cheb.lagrangePolInt(temp0, jj, chebRts, md.pp1)*temp1

        # Local to potential
        for jj in range(md.pp1):
            temp1 = math.exp( (md.ss/2.)*pl.stepSize*lam )
            md.ltp0[jj] = temp1*cheb.lagrangePolInt(-1., jj, chebRts, md.pp1)
        for ii in range(md.ss):
            temp0 = 2.*(ii+1)/md.ss-1.
            for jj in range(md.pp1):
                temp1 = math.exp( -(ii+1.-md.ss/2.)*pl.stepSize*lam )
                md.ltp[ii*md.pp1+jj] = cheb.lagrangePolInt(temp0, jj, chebRts, md.pp1)*temp1

        # Moment to local coefficients
        for ll in range(md.nlevsCut):
            for ii in range(md.pp1):
                for jj in range(md.pp1):
                    temp2 = md.kl[md.nlevs-ll]*0.5*(chebRts[jj]-chebRts[ii])*md.ss*pl.stepSize
                    temp0 = md.kl[md.nlevs-ll]*(2.+0.5*(chebRts[jj]-chebRts[ii]))*md.ss*pl.stepSize
                    kFL(&temp0, kFLP, &temp1)
                    md.mtl[2*ll*md.pp1**2+ii*md.pp1+jj] = math.exp(temp1 + temp2*lam)
                    temp0 = md.kl[md.nlevs-ll]*(3.+0.5*(chebRts[jj]-chebRts[ii]))*md.ss*pl.stepSize
                    kFL(&temp0, kFLP, &temp1)
                    md.mtl[(2*ll+1)*md.pp1**2+ii*md.pp1+jj] = math.exp(temp1 + temp2*lam)
                    
        free(chebRts)

    return 0


# Execute FMM
cdef int execute_conv_fmmExpCheb(conv_plan* pl, double* dataIn, double* dataOut) nogil except -1:

    cdef:
        int ii, jj, kk, ll, mm, nn
        double* moments0
        double* moments1
        double* local
        int orderM1Half
        double temp0, signData, signKernel
        methodData_fmmExpCheb* md
    
    # Helper stuff
    orderM1Half = <int> ((pl.order-1)/2)
    md = <methodData_fmmExpCheb*> pl.methodData
    
    # Symmetries
    signData = base.symSignFac(pl.symData)
    signKernel = base.symSignFac(pl.symKernel)

    if md.longRangeFlag == 1:
        # Memory allocation
        moments0 = <double*> calloc(md.kTotalCut*md.pp1, sizeof(double))
        moments1 = <double*> calloc(md.kTotalCut*md.pp1, sizeof(double))
        local = <double*> calloc(md.kTotalCut*md.pp1, sizeof(double))
        if (NULL == moments0 or NULL == moments1 or NULL == local):
            free(moments0)
            free(moments1)
            free(local)
            with gil:        
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

        ## One exponential scaling of moments
        # Finest level moment calculation
        mm = <int> (pl.nData-2)/md.ss
        blas.dgemm('n', 'n', &md.pp1, &mm, &md.ss, &ONED, md.stm, &md.pp1, 
                   &dataIn[orderM1Half+1], &md.ss, &ZEROD, moments0, &md.pp1)
        for ii in range(mm*md.ss+1, pl.nData):
            nn = (ii-1) - mm*md.ss
            for jj in range(md.pp1):
                moments0[mm*md.pp1+jj] += md.stm[nn*md.pp1+jj]*dataIn[orderM1Half+ii]

        # Upward Pass / Moment to moment
        mm = 2*md.pp1
        for ll in range(1, md.nlevsCut):
            blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.mtmm[(ll-1)*md.pp1**2], &md.pp1, 
                       &moments0[md.klCum[ll-1]*md.pp1], &mm, &ZEROD, &moments0[md.klCum[ll]*md.pp1], &md.pp1)
            blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.mtmp[(ll-1)*md.pp1**2], &md.pp1, 
                       &moments0[(md.klCum[ll-1]+1)*md.pp1], &mm, &ONED, &moments0[md.klCum[ll]*md.pp1], &md.pp1)

        ## Other exponential scaling of moments
        # Finest level moment calculation
        mm = <int> (pl.nData-2)/md.ss
        blas.dgemm('n', 'n', &md.pp1, &mm, &md.ss, &ONED, md.ltp, &md.pp1, 
                   &dataIn[orderM1Half+1], &md.ss, &ZEROD, moments1, &md.pp1)
        for ii in range(mm*md.ss+1, pl.nData):
            nn = (ii-1) - mm*md.ss
            for jj in range(md.pp1):
                moments1[mm*md.pp1+jj] += md.ltp[nn*md.pp1+jj]*dataIn[orderM1Half+ii]

        # Upward Pass / Moment to moment
        mm = 2*md.pp1
        for ll in range(1, md.nlevsCut):
            blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.ltlm[(ll-1)*md.pp1**2], &md.pp1, 
                       &moments1[md.klCum[ll-1]*md.pp1], &mm, &ZEROD, &moments1[md.klCum[ll]*md.pp1], &md.pp1)
            blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.ltlp[(ll-1)*md.pp1**2], &md.pp1, 
                       &moments1[(md.klCum[ll-1]+1)*md.pp1], &mm, &ONED, &moments1[md.klCum[ll]*md.pp1], &md.pp1)

        ## Use the moments for one kernel side
        # Interaction Phase / Moment to local
        for ll in range(md.nlevsCut):
            # Left kernel right axis
            for kk in range(3, md.kl[ll], 2):   # If odd
                blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                           &moments0[(kk+md.klCum[ll]-2)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
                blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                           &moments0[(kk+md.klCum[ll]-3)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
            for kk in range(2, md.kl[ll], 2):   # If even    
                blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                           &moments0[(kk+md.klCum[ll]-2)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &MONE)
            # Left kernel left axis
            # If even (kk = 0)
            blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments1[(md.klCum[ll]+1)*md.pp1], &MONE, &ONED, &local[md.klCum[ll]*md.pp1], &ONE)
            # If odd (kk = 1)
            blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                       &moments1[(md.klCum[ll]+0)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+1)*md.pp1], &ONE)
            blas.dgemv('n', &md.pp1, &md.pp1, &signData, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                       &moments1[(md.klCum[ll]+1)*md.pp1], &MONE, &ONED, &local[(md.klCum[ll]+1)*md.pp1], &ONE)  
                    
        # Downward Pass / Local to local
        mm = 2*md.pp1
        for ll in range(md.nlevsCut-1, 0, -1):
            blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.ltlm[(ll-1)*md.pp1**2], &md.pp1, 
                       &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[md.klCum[ll-1]*md.pp1], &mm)
            blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.ltlp[(ll-1)*md.pp1**2], &md.pp1, 
                       &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[(md.klCum[ll-1]+1)*md.pp1], &mm)

        # Potential evaluation / local to potential
        mm = (pl.nDataOut-2)/md.ss
        blas.dgemm('t', 'n', &md.ss, &mm, &md.pp1, &ONED, md.ltp, &md.pp1, local, &md.pp1, &ONED, &dataOut[1], &md.ss)
        for ii in range(mm*md.ss+1, pl.nDataOut):
            nn = (ii-1) - mm*md.ss
            for jj in range(md.pp1):
                dataOut[ii] += md.ltp[nn*md.pp1+jj]*local[mm*md.pp1+jj]
        for jj in range(md.pp1):
            dataOut[0] += local[jj]*md.ltp0[jj]

        ## Use the moments for other kernel side
        memset(local, 0, md.kTotalCut*md.pp1*sizeof(double))
        # Interaction Phase / Moment to local
        for ll in range(md.nlevsCut):
            # Right kernel
            for kk in range(0, md.kl[ll]-2, 2): # If even
                blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                           &moments1[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
                blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll+1)*md.pp1**2], &md.pp1, 
                           &moments1[(kk+md.klCum[ll]+3)*md.pp1], &ONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
            for kk in range(1, md.kl[ll]-2, 2): # If odd     
                blas.dgemv('t', &md.pp1, &md.pp1, &signKernel, &md.mtl[(2*ll)*md.pp1**2], &md.pp1, 
                           &moments1[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
                                          
        # Downward Pass / Local to local
        mm = 2*md.pp1
        for ll in range(md.nlevsCut-1, 0, -1):
            blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.mtmm[(ll-1)*md.pp1**2], &md.pp1, 
                       &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[md.klCum[ll-1]*md.pp1], &mm)
            blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, &md.mtmp[(ll-1)*md.pp1**2], &md.pp1, 
                       &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[(md.klCum[ll-1]+1)*md.pp1], &mm)

        # Potential evaluation / local to potential
        mm = (pl.nDataOut-2)/md.ss
        blas.dgemm('t', 'n', &md.ss, &mm, &md.pp1, &ONED, md.stm, &md.pp1, local, &md.pp1, &ONED, &dataOut[1], &md.ss)
        for ii in range(mm*md.ss+1, pl.nDataOut):
            nn = (ii-1) - mm*md.ss
            for jj in range(md.pp1):
                dataOut[ii] += md.stm[nn*md.pp1+jj]*local[mm*md.pp1+jj]
        for jj in range(md.pp1):
            dataOut[0] += local[jj]*md.stm0[jj]

        free(moments0)
        free(moments1)
        free(local)

    # Direct short range
    for ii in range(1,pl.nData):
        kk = (ii-1)/md.ss
        mm = min(pl.nData-ii, (kk+2)*md.ss-ii+1)
        for jj in range(mm):
            dataOut[ii] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii+jj]

    for jj in range(1, min(2*md.ss+1,pl.nData)):
        dataOut[0] += signKernel*pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+jj]
    dataOut[0] += (signData+signKernel)*0.5*pl.kernel[orderM1Half]*dataIn[orderM1Half]

    for ii in range(1,pl.nDataOut):
        kk = (ii-1)/md.ss
        mm = ii-(kk-1)*md.ss
        for jj in range(max(1,1+ii-pl.nData),min(mm,ii)):
            dataOut[ii] += pl.kernel[orderM1Half+jj]*dataIn[orderM1Half+ii-jj]

    for ii in range(md.ss+1):
         for jj in range(1,md.ss+1):
            dataOut[ii] += signData*pl.kernel[orderM1Half+ii+jj]*dataIn[orderM1Half+jj]
    for ii in range(1,pl.nDataOut):
        dataOut[ii] += (1.+signData)*0.5*pl.kernel[orderM1Half+ii]*dataIn[orderM1Half]

    return 0


# Destroy FMM
cdef int destroy_conv_fmmExpCheb(conv_plan* pl) nogil except -1:

    cdef:
        methodData_fmmExpCheb* md
        
    if not NULL == pl:
        md = <methodData_fmmExpCheb*> pl.methodData
        free(md.kl)
        free(md.klCum)
        free(md.mtmp)
        free(md.mtmm)
        free(md.mtl)
        free(md.ltp)
        free(md.ltp0)
        free(md.stm)
        free(md.stm0)
        free(md.ltlp)
        free(md.ltlm)
        free(md)
            
    return 0

##############

cdef int kernelFunExpHelper(double* xx, void* par, double* out) nogil:

    cdef:
        interp.Interpolator* kernelInterp = <interp.Interpolator*> par
        
    kernelInterp.interpolate(kernelInterp, xx, out)
    out[0] = math.exp(out[0])
    
    return 0


ctypedef struct kernelFunCombinerStruct:
    double tMeanMTauMean
    double delta
    double lam
    funPtr kernelFunLog
    void* kernelFunLogPar

cdef int kernelFunCombiner(double* xx, void* par, double* out) nogil:

    cdef:
        kernelFunCombinerStruct* st = <kernelFunCombinerStruct*> par
        double temp0, temp1, temp2
    
    temp0 = st.tMeanMTauMean+st.delta*xx[0]
    st.kernelFunLog(&temp0, st.kernelFunLogPar, &temp1)
    temp0 = st.tMeanMTauMean
    st.kernelFunLog(&temp0, st.kernelFunLogPar, &temp2)
    out[0] = math.exp( temp1 - temp2 + st.lam*st.delta*xx[0] )
        
    return 0
    
#############################################################################################################################################



cdef int _linReg(double* vals, int nn, double* aa, double* bb) nogil except -1:

    cdef:
        int ii
        double* ab = [0., 0.]
        double* xtx = [nn, (nn-1.)*nn/2., (nn-1.)*nn/2., (nn-1.)*nn*(2.*nn-1.)/6.]
        double* xtvals = [0., 0.]
    
    if nn < 2:
        with gil:
            raise ValueError("Number of values too small for linear regression")
    
    for ii in range(nn):
        xtvals[0] += vals[ii]
        xtvals[1] += ii*vals[ii]
    
    bb[0] = (xtvals[1] - xtx[2]/xtx[0]*xtvals[0])/(xtx[3]-xtx[2]/xtx[0]*xtx[1])
    aa[0] = (xtvals[0] - xtx[1]*bb[0])/xtx[0]

    return 0


############################################################################################################################################


# Smoothing function
cdef double _smoother(double xx) nogil:
    if xx <= 0.:
        return 1.
    elif (xx-1.) >= 0.:
        return 0.
    else:
        return 1./(math.exp(1./(1.-xx)-1./xx)+1.)
        









