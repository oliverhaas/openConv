

import numpy
import os.path
import datetime

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

cimport scipy.linalg.cython_blas as blas

cimport openAbel.mathFun as mathFun
cimport openAbel.constants as constants
from openAbel.abel.base cimport abel_plan


############################################################################################################################################
### Fast multipole method trapezoidal with end corrections                                                                               ###


ctypedef struct methodData_FMM:
    double* chebRoots
    int* kl
    int* klCum
    double* mtmp
    double* mtmm
    double* ltp
    double* mtlk
    double* direct
    double* direct0
    int pp
    int pp1
    int ss    
    int nlevs
    int kTotal
    double* coeffsSing
    double* coeffsNonsing
    int order
    double* coeffsFilter
    int orderFilter


# Plan FMM
# Eps 1.e-15 is enough for machine precision error (caused by FMM)
cdef int plan_fat_fmmTrapEndCorr(abel_plan* pl, int order = 2, double eps = 1.e-15) nogil except -1:

    cdef:
        int ii, jj, ll, kk, mm
        methodData_FMM* md
        double ti, tauj, temp
        double[:,::1] coeffs_sing_small_memView
        double[:,::1] coeffs_sing_large_memView
        double[:,::1] coeffs_nonsing_sqrt_small_memView
        double[:,::1] coeffs_nonsing_sqrt_large_memView
        double[::1] coeffs_filter_memView
        int orderM1Half, orderM1HalfInner
        double nInvSca, yInvSca
        int yCross, yLarge, yInvScaInt, nCross, nLarge, nInvScaInt

    # Input check
    if NULL == pl or order <= 0 or eps <= 1.e-18:
        with gil:
            raise ValueError('Illegal input argument.')   

    # Main method struct
    md = <methodData_FMM*> malloc(sizeof(methodData_FMM))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.chebRoots = NULL
        md.kl = NULL
        md.klCum = NULL
        md.mtmp = NULL
        md.mtmm = NULL
        md.mtlk = NULL
        md.direct = NULL
        md.coeffsSing = NULL
        md.coeffsNonsing = NULL
        md.coeffsFilter = NULL
    pl.methodData = <void*> md

    # Small data set
    if pl.nData < order+2:
        destroy_fat_fmmTrapEndCorr(pl)
        with gil:
            raise ValueError('Not enough data points for given parameters.')

    # Load and prepare end correction coefficients
    md.order = order
    orderM1Half = <int> ((md.order-1)/2)
    orderM1HalfInner = <int> (md.order/2)
    md.coeffsSing = <double*> malloc(md.order*(pl.nData-1)*sizeof(double))
    md.coeffsNonsing = <double*> malloc(md.order*(pl.nData-1)*sizeof(double))
    if (NULL == md.coeffsSing or NULL == md.coeffsNonsing):
        destroy_fat_fmmTrapEndCorr(pl)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    if pl.forwardBackward == -1:    # Forward transform
        with gil:
            try:
                coeffs_sing_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqLin_sing_large_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_small_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_large_" + "%02d" % order + ".npy")
                if pl.shift == 0.:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqLin_sing_small_" + "%02d" % order + ".npy")
                elif pl.shift == 0.5:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqLin_sing_small_halfShift_" + "%02d" % order + ".npy")
                else:
                    raise NotImplementedError('Method not implemented for given parameters.')
            except:
                destroy_fat_fmmTrapEndCorr(pl)
                raise
        yCross = coeffs_sing_small_memView.shape[0]
        yLarge = coeffs_sing_large_memView.shape[0]
        for ii in range(min(yCross,pl.nData-1)):
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = coeffs_sing_small_memView[ii,jj]
        for ii in range(yCross, pl.nData-1):
            yInvSca = pl.stepSize/pl.grid[ii]*(yCross-1)*(yLarge-1)
            yInvScaInt = <int> mathFun.fmax(mathFun.fmin(yInvSca,yLarge-3),1)
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = interpCubic(yInvSca-yInvScaInt, md.order, &coeffs_sing_large_memView[yInvScaInt-1,jj]) * \
                                                mathFun.sqrt(pl.grid[ii]/2./pl.stepSize)
        nCross = coeffs_nonsing_sqrt_small_memView.shape[0]            
        for ii in range(max(pl.nData-1-nCross,0),pl.nData-1):
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = coeffs_nonsing_sqrt_small_memView[pl.nData-2-ii,jj]*(pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize) / \
                                                   mathFun.sqrt((pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii])*(pl.grid[pl.nData-1]-pl.grid[ii]))
        nLarge = coeffs_nonsing_sqrt_large_memView.shape[0]      
        for ii in range(max(pl.nData-1-nCross,0)):
            nInvSca = pl.stepSize/(pl.grid[pl.nData-1]-pl.grid[ii])*nCross*(nLarge-1)
            nInvScaInt = <int> mathFun.fmax(mathFun.fmin(nInvSca,nLarge-3),1)
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = interpCubic(nInvSca-nInvScaInt, md.order, &coeffs_nonsing_sqrt_large_memView[nInvScaInt-1,jj]) * \
                                                   (pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize) / \
                                                   mathFun.sqrt((pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii])*(pl.grid[pl.nData-1]-pl.grid[ii]))
        for ii in range(pl.nData-1):
            md.coeffsNonsing[md.order*ii+orderM1HalfInner] -= 0.5*pl.grid[pl.nData-1]/mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)

    elif pl.forwardBackward == 1 or pl.forwardBackward == 2:    # Backward transform
        with gil:
            try:
                coeffs_sing_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSq_sing_large_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_small_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_large_" + "%02d" % order + ".npy")
                if pl.shift == 0.:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSq_sing_small_" + "%02d" % order + ".npy")
                elif pl.shift == 0.5:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSq_sing_small_halfShift_" + "%02d" % order + ".npy")
                else:
                    raise NotImplementedError('Method not implemented for given parameters.')
            except:
                destroy_fat_fmmTrapEndCorr(pl)
                raise
        yCross = coeffs_sing_small_memView.shape[0]
        yLarge = coeffs_sing_large_memView.shape[0]
        for ii in range(min(yCross,pl.nData-1)):
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = coeffs_sing_small_memView[ii,jj]/pl.stepSize
        for ii in range(yCross, pl.nData-1):
            yInvSca = pl.stepSize/pl.grid[ii]*(yCross-1)*(yLarge-1)
            yInvScaInt = <int> mathFun.fmax(mathFun.fmin(yInvSca,yLarge-3),1)
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = interpCubic(yInvSca-yInvScaInt, md.order, &coeffs_sing_large_memView[yInvScaInt-1,jj]) / \
                                                mathFun.sqrt(pl.grid[ii]*2.*pl.stepSize)
        nCross = coeffs_nonsing_sqrt_small_memView.shape[0]            
        for ii in range(max(pl.nData-1-nCross,0),pl.nData-1):
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = coeffs_nonsing_sqrt_small_memView[pl.nData-2-ii,jj] / \
                                                   mathFun.sqrt((pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii])*(pl.grid[pl.nData-1]-pl.grid[ii]))
        nLarge = coeffs_nonsing_sqrt_large_memView.shape[0]      
        for ii in range(max(pl.nData-1-nCross,0)):
            nInvSca = pl.stepSize/(pl.grid[pl.nData-1]-pl.grid[ii])*nCross*(nLarge-1)
            nInvScaInt = <int> mathFun.fmax(mathFun.fmin(nInvSca,nLarge-3),1)
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = interpCubic(nInvSca-nInvScaInt, md.order, &coeffs_nonsing_sqrt_large_memView[nInvScaInt-1,jj]) / \
                                                   mathFun.sqrt((pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii])*(pl.grid[pl.nData-1]-pl.grid[ii]))
        for ii in range(pl.nData-1):
            md.coeffsNonsing[md.order*ii+orderM1HalfInner] -= 0.5/mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)

    elif pl.forwardBackward == -2:    # Modified forward transform for 1/r^2 singular functions
        with gil:
            try:
                coeffs_sing_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqY2OR2_sing_large_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_small_" + "%02d" % order + ".npy")
                coeffs_nonsing_sqrt_large_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrt_nonsing_large_" + "%02d" % order + ".npy")
                if pl.shift == 0.:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqY2OR2_sing_small_" + "%02d" % order + ".npy")
                elif pl.shift == 0.5:
                    coeffs_sing_small_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_invSqrtDiffSqY2OR2_sing_small_halfShift_" + "%02d" % order + ".npy")
                else:
                    raise NotImplementedError('Method not implemented for given parameters.')
            except:
                destroy_fat_fmmTrapEndCorr(pl)
                raise
        yCross = coeffs_sing_small_memView.shape[0]
        yLarge = coeffs_sing_large_memView.shape[0]
        for ii in range(min(yCross,pl.nData-1)):
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = coeffs_sing_small_memView[ii,jj]/pl.stepSize
        for ii in range(yCross, pl.nData-1):
            yInvSca = pl.stepSize/pl.grid[ii]*(yCross-1)*(yLarge-1)
            yInvScaInt = <int> mathFun.fmax(mathFun.fmin(yInvSca,yLarge-3),1)
            for jj in range(md.order):
                md.coeffsSing[md.order*ii+jj] = interpCubic(yInvSca-yInvScaInt, md.order, &coeffs_sing_large_memView[yInvScaInt-1,jj]) / \
                                                mathFun.sqrt(pl.grid[ii]*2.*pl.stepSize)
        nCross = coeffs_nonsing_sqrt_small_memView.shape[0]            
        for ii in range(max(pl.nData-1-nCross,0),pl.nData-1):
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = coeffs_nonsing_sqrt_small_memView[pl.nData-2-ii,jj] * \
                                                   (pl.grid[ii]/(pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize))**2 / \
                                                   mathFun.sqrt( (pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii]) *
                                                                 (pl.grid[pl.nData-1]-pl.grid[ii]) )
        nLarge = coeffs_nonsing_sqrt_large_memView.shape[0]      
        for ii in range(max(pl.nData-1-nCross,0)):
            nInvSca = pl.stepSize/(pl.grid[pl.nData-1]-pl.grid[ii])*nCross*(nLarge-1)
            nInvScaInt = <int> mathFun.fmax(mathFun.fmin(nInvSca,nLarge-3),1)
            for jj in range(md.order):
                md.coeffsNonsing[md.order*ii+jj] = interpCubic(nInvSca-nInvScaInt, md.order, &coeffs_nonsing_sqrt_large_memView[nInvScaInt-1,jj]) * \
                                                   (pl.grid[ii]/(pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize))**2 / \
                                                   mathFun.sqrt( (pl.grid[pl.nData-1]+(jj-orderM1HalfInner)*pl.stepSize+pl.grid[ii]) *
                                                                 (pl.grid[pl.nData-1]-pl.grid[ii]) )
        for ii in range(pl.nData-1):
            md.coeffsNonsing[md.order*ii+orderM1HalfInner] -= 0.5*(pl.grid[ii]/pl.grid[pl.nData-1])**2/mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)

    else:
        destroy_fat_fmmTrapEndCorr(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Hierarchical decomposition
#    md.pp = max(md.order+1, <int> ( -0.7*mathFun.log(eps)/mathFun.log(5.) + 1. ) )      # Somewhat empiral scaling constant
    md.pp = max(4, <int> ( -0.55*mathFun.log(2.*eps) + 1. ) )                           # New empirical scaling, should be exponential
    md.pp1 = md.pp + 1
    md.nlevs = max(<int> ( mathFun.log2((pl.nData-1.)/(2.*md.pp)) + 1. ), 2)
    md.ss = max(<int> ( (pl.nData-1.)/2**md.nlevs + 1. ), 3)                            # ss ~= 2*pp theoretical
    md.kTotal = 2**(md.nlevs+1) - 1                                                     # Total number of intervals in all levels
    
    # Allocation of arrays FMM part
    md.chebRoots = <double*> malloc(md.pp1*sizeof(double))
    md.kl = <int*> malloc((md.nlevs+1)*sizeof(int))
    md.klCum = <int*> malloc((md.nlevs+1)*sizeof(int))
    md.mtmp = <double*> malloc(md.pp1**2*sizeof(double))
    md.mtmm = <double*> malloc(md.pp1**2*sizeof(double))
    md.ltp = <double*> malloc(md.pp1*md.ss*sizeof(double))
    md.mtlk = <double*> calloc(2*md.kTotal*md.pp1**2, sizeof(double))
    md.direct = <double*> calloc(2**md.nlevs*md.ss**2*2, sizeof(double))
    md.direct0 = <double*> malloc(2*md.ss*sizeof(double))

    if (NULL == md.chebRoots or NULL == md.kl or NULL == md.klCum or NULL == md.mtmp or NULL == md.mtmm or 
        NULL == md.ltp or NULL == md.mtlk or NULL == md.direct or NULL == md.direct0):
        destroy_fat_fmmTrapEndCorr(pl)        
        with gil:        
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

    # More hierarchical decomposition stuff
    md.klCum[0] = 0
    md.kl[0] = 2**md.nlevs
    for ii in range(1,md.nlevs+1):
        md.kl[ii] = 2**(md.nlevs-ii)
        md.klCum[ii] = md.klCum[ii-1] + md.kl[ii-1]
    _chebRoots(md.pp1, md.chebRoots)

    # Moment to moment coefficients
    for ii in range(md.pp1):
        for jj in range(md.pp1):
            md.mtmp[ii*md.pp1+jj] = _lagrangePInt(0.5*md.chebRoots[jj]+0.5, ii, md.chebRoots, md.pp1)
            md.mtmm[ii*md.pp1+jj] = _lagrangePInt(0.5*md.chebRoots[jj]-0.5, ii, md.chebRoots, md.pp1)    

    # Local to potential coefficients
    for ii in range(md.ss):
        temp = 2.*(ii+1)/md.ss-1.
        for jj in range(md.pp1):
            md.ltp[ii*md.pp1+jj] = _lagrangePInt(temp, jj, md.chebRoots, md.pp1)

    # Different kernels for forward and backward transform
    if pl.forwardBackward == -1:
        # Moment to local coefficients
        for ll in range(md.nlevs-1):
            # If even
            for kk in range(0, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift
                    for jj in range(md.pp1):
                        tauj = md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = tauj/mathFun.sqrt(tauj**2-ti**2)
                        tauj = md.kl[md.nlevs-ll]*(kk+3.5+0.5*md.chebRoots[jj])*md.ss + pl.shift
                        md.mtlk[(2*(md.klCum[ll]+kk)+1)*md.pp1**2+ii*md.pp1+jj] = tauj/mathFun.sqrt(tauj**2-ti**2)
            # If odd
            for kk in range(1, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift   
                    for jj in range(md.pp1):
                        tauj = md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = tauj/mathFun.sqrt(tauj**2-ti**2)   

        # Direct short range coefficients
        for ii in range(1, pl.nData):
            kk = <int> ((ii-1)/md.ss)
            ll = (ii-1) - kk*md.ss
            mm = min(pl.nData-kk*md.ss-1, 2*md.ss)
            for jj in range(ll+1, mm):
                md.direct[kk*md.ss**2*2+md.ss*2*ll+jj] = pl.grid[ii+jj-ll]/mathFun.sqrt(pl.grid[ii+jj-ll]**2 - pl.grid[ii]**2)
        mm = min(pl.nData, 2*md.ss+1)
        for jj in range(1, mm):
            md.direct0[jj-1] = pl.grid[jj]/mathFun.sqrt(pl.grid[jj]**2 - pl.grid[0]**2)
    elif pl.forwardBackward == -2:
        # Moment to local coefficients
        for ll in range(md.nlevs-1):
            # If even
            for kk in range(0, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = (md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift)*pl.stepSize
                    for jj in range(md.pp1):
                        tauj = (md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = (ti/tauj)**2/mathFun.sqrt(tauj**2-ti**2)
                        tauj = (md.kl[md.nlevs-ll]*(kk+3.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk)+1)*md.pp1**2+ii*md.pp1+jj] = (ti/tauj)**2/mathFun.sqrt(tauj**2-ti**2)
            # If odd
            for kk in range(1, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = (md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift)*pl.stepSize
                    for jj in range(md.pp1):
                        tauj = (md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = (ti/tauj)**2/mathFun.sqrt(tauj**2-ti**2)   

        # Direct short range coefficients
        for ii in range(1, pl.nData):
            kk = <int> ((ii-1)/md.ss)
            ll = (ii-1) - kk*md.ss
            mm = min(pl.nData-kk*md.ss-1, 2*md.ss)
            for jj in range(ll+1, mm):
                md.direct[kk*md.ss**2*2+md.ss*2*ll+jj] = (pl.grid[ii]/pl.grid[ii+jj-ll])**2/mathFun.sqrt(pl.grid[ii+jj-ll]**2 - pl.grid[ii]**2)
        mm = min(pl.nData, 2*md.ss+1)
        for jj in range(1, mm):
            md.direct0[jj-1] = (pl.grid[0]/pl.grid[jj])**2/mathFun.sqrt(pl.grid[jj]**2 - pl.grid[0]**2)    
    elif pl.forwardBackward == 1 or pl.forwardBackward == 2:
        # Moment to local coefficients
        for ll in range(md.nlevs-1):
            # If even
            for kk in range(0, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = (md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift)*pl.stepSize
                    for jj in range(md.pp1):
                        tauj = (md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = 1./mathFun.sqrt(tauj**2-ti**2)
                        tauj = (md.kl[md.nlevs-ll]*(kk+3.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk)+1)*md.pp1**2+ii*md.pp1+jj] = 1./mathFun.sqrt(tauj**2-ti**2)
            # If odd
            for kk in range(1, md.kl[ll]-2, 2):
                for ii in range(md.pp1):
                    ti = (md.kl[md.nlevs-ll]*(kk+0.5+0.5*md.chebRoots[ii])*md.ss + pl.shift)*pl.stepSize
                    for jj in range(md.pp1):
                        tauj = (md.kl[md.nlevs-ll]*(kk+2.5+0.5*md.chebRoots[jj])*md.ss + pl.shift)*pl.stepSize
                        md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2+ii*md.pp1+jj] = 1./mathFun.sqrt(tauj**2-ti**2)   

        # Direct short range coefficients
        for ii in range(1, pl.nData):
            kk = <int> ((ii-1)/md.ss)
            ll = (ii-1) - kk*md.ss
            mm = min(pl.nData-kk*md.ss-1, 2*md.ss)
            for jj in range(ll+1, mm):
                md.direct[kk*md.ss**2*2+md.ss*2*ll+jj] = 1./mathFun.sqrt(pl.grid[ii+jj-ll]**2 - pl.grid[ii]**2)
        mm = min(pl.nData, 2*md.ss+1)
        for jj in range(1, mm):
            md.direct0[jj-1] = 1./mathFun.sqrt(pl.grid[jj]**2 - pl.grid[0]**2)    
    else:
        destroy_fat_fmmTrapEndCorr(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Input modification filter
    if pl.forwardBackward == 1:
        md.orderFilter = md.order+1 + (md.order % 2)
        md.coeffsFilter = <double*> malloc(md.orderFilter*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_fmmTrapEndCorr(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        with gil:
            try:
                coeffs_filter_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_deriv_smooth_" + "%02d" % (md.orderFilter-1) + ".npy")
            except:
                destroy_fat_fmmTrapEndCorr(pl)
                raise
        for ii in range(md.orderFilter):
            md.coeffsFilter[ii] = coeffs_filter_memView[ii]*(-constants.piinv)
    elif pl.forwardBackward == 2 or pl.forwardBackward == -1 or pl.forwardBackward == -2:
        md.orderFilter = 1
        md.coeffsFilter = <double*> malloc(1*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_fmmTrapEndCorr(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        if pl.forwardBackward == 2:            
            md.coeffsFilter[0] = -constants.piinv*pl.stepSize
        else:
            md.coeffsFilter[0] = 2.*pl.stepSize
    else:
        destroy_fat_fmmTrapEndCorr(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    return 0


# Execute FMM
cdef int execute_fat_fmmTrapEndCorr(abel_plan* pl, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1:

    cdef:
        methodData_FMM* md
        int ii, jj, kk, ll, mm, nn
        double* moments = NULL
        double* local = NULL
        double* dataInTemp0 = NULL
        double* dataInTemp1 = NULL
        int orderM1Half, orderFilterM1Half, orderM1HalfInner
        int nLeftExt, nRightExt

    md = <methodData_FMM*> pl.methodData
    orderM1Half = <int> ((md.order-1)/2)
    orderM1HalfInner = <int> (md.order/2)
    orderFilterM1Half = <int> ((md.orderFilter-1)/2)

    # Allocate temporary data arrays
    dataInTemp0 = <double*> malloc((pl.nData+md.order+md.orderFilter-2)*sizeof(double))
    dataInTemp1 = <double*> malloc((pl.nData+md.order-1)*sizeof(double))
    if NULL == dataInTemp0 or NULL == dataInTemp1:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    
    # Left boundary handling
    if leftBoundary == 0 or leftBoundary == 1 or leftBoundary == 2:
        nLeftExt = orderM1Half + orderFilterM1Half
    elif leftBoundary == 3:
        nLeftExt = 0
    else:
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
    # Right boundary handling
    if rightBoundary == 0: # TODO maybe or rightBoundary == 1 or rightBoundary == 2:
        nRightExt = orderM1Half + orderFilterM1Half
    elif rightBoundary == 3:
        nRightExt = 0
    else:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')           
    # Copy and extend data if necessary
    nn = max(md.order, md.orderFilter-1)
    for ii in range(pl.nData+md.order+md.orderFilter-2-nLeftExt-nRightExt):
        dataInTemp0[nLeftExt+ii] = dataIn[ii]
    if leftBoundary == 0:
        for ii in range(nLeftExt):
            dataInTemp0[ii] = polint(&dataInTemp0[nLeftExt], nn, ii-nLeftExt)
    elif leftBoundary == 1:
        if pl.shift == 0.:
            for ii in range(nLeftExt):
                dataInTemp0[nLeftExt-1-ii] = -dataInTemp0[nLeftExt+1+ii]
        elif pl.shift == 0.5:
            for ii in range(nLeftExt):
                dataInTemp0[nLeftExt-1-ii] = -dataInTemp0[nLeftExt+ii]
        else:
            free(dataInTemp0)
            free(dataInTemp1)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif leftBoundary == 2:
        if pl.shift == 0.:
            for ii in range(nLeftExt):
                dataInTemp0[nLeftExt-1-ii] = dataInTemp0[nLeftExt+1+ii]
        elif pl.shift == 0.5:
            for ii in range(nLeftExt):
                dataInTemp0[nLeftExt-1-ii] = dataInTemp0[nLeftExt+ii]
        else:
            free(dataInTemp0)
            free(dataInTemp1)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif leftBoundary == 3:
        pass
    else:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
    if rightBoundary == 0:
        for ii in range(nRightExt):
            dataInTemp0[pl.nData+orderM1Half+orderFilterM1Half+ii] = polint(&dataInTemp0[pl.nData+orderM1Half+orderFilterM1Half-nn], nn, ii+nn)
    elif rightBoundary == 3:
        pass
    else:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Do scaling or numerical derivative
    convolve(dataInTemp0, pl.nData+md.order-1, dataInTemp1, md.orderFilter, md.coeffsFilter)
    free(dataInTemp0)

    # Allocate temporary data arrays
    moments = <double*> calloc(md.kTotal*md.pp1, sizeof(double))
    local = <double*> calloc(md.kTotal*md.pp1, sizeof(double))
    if moments == NULL or local == NULL:
        free(moments)
        free(local)
        free(dataInTemp1)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')

    # Finest level moment calculation
    mm = (pl.nData-2)/md.ss - 2         # basically kl[0]-2, as first two block are not needed
    blas.dgemm('n', 'n', &md.pp1, &mm, &md.ss, &ONED, md.ltp, &md.pp1, 
               &dataInTemp1[orderM1Half+2*md.ss+1], &md.ss, &ZEROD, &moments[2*md.pp1], &md.pp1)
    mm += 2
    for ii in range(mm*md.ss+1, pl.nData):
        nn = (ii-1) - mm*md.ss
        for jj in range(md.pp1):
            moments[mm*md.pp1+jj] += md.ltp[nn*md.pp1+jj]*dataInTemp1[orderM1Half+ii]

    # Upward Pass / Moment to moment
    mm = 2*md.pp1
    for ll in range(1, md.nlevs-1):
        blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmm, &md.pp1, 
                   &moments[md.klCum[ll-1]*md.pp1], &mm, &ONED, &moments[md.klCum[ll]*md.pp1], &md.pp1)
        blas.dgemm('t', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmp, &md.pp1, 
                   &moments[(md.klCum[ll-1]+1)*md.pp1], &mm, &ONED, &moments[md.klCum[ll]*md.pp1], &md.pp1)

    # Interaction Phase / Moment to local
    for ll in range(md.nlevs-1):
        # If even
        for kk in range(0, md.kl[ll]-2, 2):
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtlk[(2*(md.klCum[ll]+kk)+1)*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+3)*md.pp1], &ONE, &ONED, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
        # If odd
        for kk in range(1, md.kl[ll]-2, 2):
            blas.dgemv('t', &md.pp1, &md.pp1, &ONED, &md.mtlk[(2*(md.klCum[ll]+kk))*md.pp1**2], &md.pp1, 
                       &moments[(kk+md.klCum[ll]+2)*md.pp1], &ONE, &ZEROD, &local[(md.klCum[ll]+kk)*md.pp1], &ONE)
    
    # Downward Pass / Local to local
    mm = 2*md.pp1
    for ll in range(md.nlevs-2, 0, -1):
        blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmm, &md.pp1, 
                   &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[md.klCum[ll-1]*md.pp1], &mm)
        blas.dgemm('n', 'n', &md.pp1, &md.kl[ll], &md.pp1, &ONED, md.mtmp, &md.pp1, 
                   &local[md.klCum[ll]*md.pp1], &md.pp1, &ONED, &local[(md.klCum[ll-1]+1)*md.pp1], &mm)

    # Set output to zero first
    memset(dataOut, 0, pl.nData*sizeof(double))

    # Potential evaluation / local to potential
    mm = (pl.nData-2)/md.ss - 1          # basically kl[0]-2, as last two block are not needed
    blas.dgemm('t', 'n', &md.ss, &mm, &md.pp1, &ONED, md.ltp, &md.pp1, local, &md.pp1, &ZEROD, &dataOut[1], &md.ss) # TODO segfau. possible?
    # TODO Shouldn't be here some more local to potential?
    for jj in range(md.pp1):
        dataOut[0] += local[jj]*_lagrangePInt(-1., jj, md.chebRoots, md.pp1)
    
    free(moments)
    free(local)

    # Direct short range
    ll = min((pl.nData-2)/md.ss, md.kl[0]-1)
    mm = 2*md.ss
    for kk in range(ll):
        blas.dgemv('t', &mm, &md.ss, &ONED, &md.direct[kk*md.ss**2*2], &mm, 
                   &dataInTemp1[orderM1Half+1+kk*md.ss], &ONE, &ONED, &dataOut[1+kk*md.ss], &ONE)
    for ii in range(ll*md.ss+1, pl.nData-1):
        kk = (ii-1)/md.ss
        nn = (ii-1) - kk*md.ss
        mm = min(pl.nData-kk*md.ss-1, 2*md.ss)
        for jj in range(nn+1, mm):
            dataOut[ii] += md.direct[kk*md.ss**2*2+md.ss*2*nn+jj]*dataInTemp1[orderM1Half+ii+jj-nn]
    mm = min(pl.nData, 2*md.ss+1)
    for jj in range(1, mm):
        dataOut[0] += md.direct0[jj-1]*dataInTemp1[orderM1Half+jj]

    # End correction left singular end
    # TODO maybe BLAS
    for ii in range(pl.nData-1):
        for jj in range(md.order):
            dataOut[ii] += md.coeffsSing[md.order*ii+jj]*dataInTemp1[ii+jj]

    # End correction right nonsingular end
    mm = pl.nData-1
    blas.dgemv('t', &md.order, &mm, &ONED, md.coeffsNonsing, &md.order, 
               &dataInTemp1[pl.nData-1+orderM1Half-orderM1HalfInner], &ONE, &ONED, dataOut, &ONE)

    free(dataInTemp1)

    return 0


# Destroy FMM
cdef int destroy_fat_fmmTrapEndCorr(abel_plan* pl) nogil except -1:

    cdef:
        methodData_FMM* md = <methodData_FMM*> pl.methodData

    free(md.chebRoots)
    free(md.kl)
    free(md.klCum)
    free(md.mtmp)
    free(md.mtmm) 
    free(md.ltp)    
    free(md.mtlk)
    free(md.direct)
    free(md.direct0)  
    free(md.coeffsSing)
    free(md.coeffsNonsing)
    free(md.coeffsFilter)
    free(md)

    return 0



#############################################################################################################################################

# Cubic interpolation
cdef inline double interpCubic(double x, int incx, double* p) nogil:

    return p[1*incx] + 0.5 * x*(p[2*incx] - p[0*incx] + 
                                x*(2.0*p[0*incx] - 5.0*p[1*incx] + 4.0*p[2*incx] - p[3*incx] + 
                                   x*(3.0*(p[1*incx] - p[2*incx]) + p[3*incx] - p[0*incx])))


# Polynomial inter-/extrapolation on equidistant grid
cdef double polint(double* data, int nData, double xx) nogil:

    cdef:
        int ii, mm, ns
        double* cc = NULL
        double* dd = NULL
        double den, res

    cc = <double*> malloc(nData*sizeof(double))
    dd = <double*> malloc(nData*sizeof(double))
    if NULL == cc or NULL == dd:
        free(cc)
        free(dd)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    ns = <int> (xx+0.5)
    ns = min(max(ns,0),nData-1)
    for ii in range(nData):
        cc[ii] = data[ii]
        dd[ii] = data[ii]

    res = data[ns]
    ns -= 1
    for mm in range(1,nData):
        for ii in range(nData-mm):
            den = (dd[ii]-cc[ii+1])/mm
            dd[ii] = (ii+mm-xx)*den
            cc[ii]= (ii-xx)*den
        if 2*(ns+1) < nData-mm:
            res += cc[ns+1]
        else:
            res += dd[ns]
            ns -= 1
    free(cc)
    free(dd)

    return res


# Apply filter; possibly just numerical derivative
cdef int convolve(double* dataIn, int nData, double* dataOut, int order, double* coeffs) nogil:

    cdef:
        int ii, jj

    memset(dataOut, 0, nData*sizeof(double))
    # TODO Maybe DGEMM or FFT here
    for ii in range(nData):
        for jj in range(order):
            dataOut[ii] += coeffs[jj]*dataIn[ii+jj]

    return 0


cdef int _chebRoots(int order, double* roots) nogil:
    
    cdef:
        int ii
    
    for ii in range(order):
        roots[ii] = mathFun.cos(0.5*constants.pi*(2.*ii+1.)/order)
    
    return 0


cdef double _lagrangePInt(double xx, unsigned int ind, double* nodes, unsigned int order) nogil:

    cdef:
        unsigned int ii
        double res
    
    if xx >= -1. and xx <= 1.:
        res = 1.
        for ii in range(ind):
            res *= (xx - nodes[ii])/(nodes[ind] - nodes[ii])
        for ii in range(ind+1, order):
            res *= (xx - nodes[ii])/(nodes[ind] - nodes[ii])
    else:
        res = 0.

    return res


cdef:
    int ONE = 1
    double ZEROD = 0.
    double ONED = 1.
    double TWOD = 2.
    double MONED = -1.
    double MTWOD = -2.




