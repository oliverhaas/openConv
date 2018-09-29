
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
### Fast Abel transforms based on trapezoidal rules                                                                                      ###
############################################################################################################################################


############################################################################################################################################
### Trapezoidal rule with constant desingularization                                                                                     ###

ctypedef struct methodData_DesingConst:
    double* desing
    double* coeffsFilter
    int orderFilter

# Plan desingularized quadrature trapezoidal
cdef int plan_fat_trapezoidalDesingConst(abel_plan* pl) nogil except -1:

    cdef:
        methodData_DesingConst* md = <methodData_DesingConst*> malloc(sizeof(methodData_DesingConst))
        int ii, jj, ll
        double[::1] coeffs_filter_memView
        double temp0, temp1
        int orderFilterM1Half

    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument.')   

    # Small data set
    if pl.nData < 3:
        with gil:
            raise ValueError('Not enough data points for given parameters.')

    # Main method struct
    md = <methodData_DesingConst*> malloc(sizeof(methodData_DesingConst))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.desing = NULL
        md.coeffsFilter = NULL
    pl.methodData = <void*> md

    # Desingularize array
    md.desing = <double*> malloc((pl.nData-1)*sizeof(double))
    if NULL == md.desing:
        destroy_fat_trapezoidalDesingConst(pl)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    if pl.forwardBackward == -1:
        for ii in range(pl.nData-1):
            temp0 = mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)
            md.desing[ii] = temp0/pl.stepSize
    elif pl.forwardBackward == 2 or pl.forwardBackward == 1:
        for ii in range(1,pl.nData-1):
            temp0 = mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)
            temp1 = mathFun.log((pl.grid[pl.nData-1]+temp0)/pl.grid[ii])
            md.desing[ii] = temp1/pl.stepSize
        temp0 = mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[0]**2)
        if pl.shift == 0.:
            md.desing[0] = 0.
        elif pl.shift == 0.5:
            temp1 = mathFun.log((pl.grid[pl.nData-1]+temp0)/pl.grid[0])
            md.desing[0] = temp1/pl.stepSize
    elif pl.forwardBackward == -2:
        for ii in range(pl.nData-1):
            md.desing[ii] = mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)/pl.grid[pl.nData-1]/pl.stepSize
    else:
        destroy_fat_trapezoidalDesingConst(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Input modification filter
    if pl.forwardBackward == 1:
        md.orderFilter = 3
        orderFilterM1Half = 1
        md.coeffsFilter = <double*> malloc(md.orderFilter*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_trapezoidalDesingConst(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        with gil:
            try:
                coeffs_filter_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_deriv_smooth_" + "%02d" % 2 + ".npy")
            except:
                destroy_fat_trapezoidalDesingConst(pl)
                raise
        for ii in range(md.orderFilter):
            md.coeffsFilter[ii] = coeffs_filter_memView[ii]*(-constants.piinv)
    elif pl.forwardBackward == 2 or pl.forwardBackward == -1 or pl.forwardBackward == -2:
        md.orderFilter = 1
        md.coeffsFilter = <double*> malloc(1*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_trapezoidalDesingConst(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        if pl.forwardBackward == 2:            
            md.coeffsFilter[0] = -constants.piinv*pl.stepSize
        else:
            md.coeffsFilter[0] = 2.*pl.stepSize
    else:
        destroy_fat_trapezoidalDesingConst(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    return 0


# Execute desingularized quadrature trapezoidal
cdef int execute_fat_trapezoidalDesingConst(abel_plan* pl, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1:

    cdef:
        int ii, jj, nn
        methodData_DesingConst* md
        double* dataInTemp0 = NULL
        double* dataInTemp1 = NULL
        int orderFilterM1Half
        int nLeftExt, nRightExt

    md = <methodData_DesingConst*> pl.methodData
    orderFilterM1Half = (md.orderFilter-1)/2

    # Allocate temporary data arrays
    dataInTemp0 = <double*> malloc((pl.nData+md.orderFilter-1)*sizeof(double))
    dataInTemp1 = <double*> malloc(pl.nData*sizeof(double))
    if NULL == dataInTemp0 or NULL == dataInTemp1:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    
    # Left boundary handling
    if leftBoundary == 0 or leftBoundary == 1 or leftBoundary == 2:
        nLeftExt = orderFilterM1Half
    elif leftBoundary == 3:
        nLeftExt = 0
    else:
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
    # Right boundary handling
    if rightBoundary == 0: # TODO or rightBoundary == 1 or rightBoundary == 2:
        nRightExt = orderFilterM1Half
    elif rightBoundary == 3:
        nRightExt = 0
    else:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')           
    # Copy and extend data if necessary
    nn = md.orderFilter-1
    for ii in range(pl.nData+md.orderFilter-1-nLeftExt-nRightExt):
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
            dataInTemp0[pl.nData+orderFilterM1Half+ii] = polint(&dataInTemp0[pl.nData+orderFilterM1Half-nn], nn, ii+nn)
    elif rightBoundary == 3:
        pass
    else:
        free(dataInTemp0)
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Do scaling or numerical derivative
    convolve(dataInTemp0, pl.nData, dataInTemp1, md.orderFilter, md.coeffsFilter)
    free(dataInTemp0)

    # Main trapezoidal rule
    memset(dataOut, 0, pl.nData*sizeof(double))
    if pl.forwardBackward == -1:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData-1):
                dataOut[ii] += (dataInTemp1[jj]-dataInTemp1[ii]) * pl.grid[jj]/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
            jj = pl.nData-1
            dataOut[ii] += 0.5*(dataInTemp1[jj]-dataInTemp1[ii]) * pl.grid[jj]/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
            dataOut[ii] += dataInTemp1[ii]*md.desing[ii]
    elif pl.forwardBackward == 1 or pl.forwardBackward == 2:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData-1):
                dataOut[ii] += (dataInTemp1[jj]-dataInTemp1[ii]) / mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
            jj = pl.nData-1
            dataOut[ii] += 0.5*(dataInTemp1[jj]-dataInTemp1[ii]) / mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
            dataOut[ii] += dataInTemp1[ii]*md.desing[ii]
    elif pl.forwardBackward == -2:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData-1):
                dataOut[ii] += (dataInTemp1[jj]-dataInTemp1[ii]) * (pl.grid[ii]/pl.grid[jj])**2/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
            jj = pl.nData-1
            dataOut[ii] += 0.5*(dataInTemp1[pl.nData-1]-dataInTemp1[ii]) * (pl.grid[ii]/pl.grid[pl.nData-1])**2/mathFun.sqrt(pl.grid[pl.nData-1]**2-pl.grid[ii]**2)
            dataOut[ii] += dataInTemp1[ii]*md.desing[ii]
    else:
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    free(dataInTemp1)

    return 0


cdef int destroy_fat_trapezoidalDesingConst(abel_plan* pl) nogil except -1:

    cdef:
        methodData_DesingConst* md = <methodData_DesingConst*> pl.methodData

    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument.')   
    free(md.desing)
    free(md.coeffsFilter)
    free(md)

    return 0


############################################################################################################################################
### Trapezoidal rule with end corrections                                                                                                ###


ctypedef struct methodData_EndCorr:
    double* coeffsSing
    double* coeffsNonsing
    double* coeffsFilter
    int order
    int orderFilter



# Plan desingularized quadrature trapezoidal
cdef int plan_fat_trapezoidalEndCorr(abel_plan* pl, int order = 2) nogil except -1:
    cdef:
        methodData_EndCorr* md
        double[:,::1] coeffs_nonsing_sqrt_small_memView
        double[:,::1] coeffs_nonsing_sqrt_large_memView
        double[:,::1] coeffs_sing_small_memView
        double[:,::1] coeffs_sing_large_memView
        double[:,::1] coeffs_ext_small_memView
        double[:,::1] coeffs_ext_large_memView
        double[::1] coeffs_filter_memView
        int ii, jj, ll
        double nInvSca, yInvSca
        int nCross, nLarge, nInvScaInt, yCross, yLarge, yInvScaInt
        int orderM1Half, orderFilterM1Half, orderM1HalfInner

    # Input check
    if NULL == pl or order <= 0:
        with gil:
            raise ValueError('Illegal input argument.')   

    # Main method struct
    md = <methodData_EndCorr*> malloc(sizeof(methodData_EndCorr))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.coeffsSing = NULL
        md.coeffsNonsing = NULL
        md.coeffsFilter = NULL
    pl.methodData = <void*> md

    # Small data set
    if pl.nData < order+2:
        destroy_fat_trapezoidalEndCorr(pl)
        with gil:
            raise ValueError('Not enough data points for given parameters.')

    # Load and prepare end correction coefficients
    md.order = order
    orderM1Half = <int> ((md.order-1)/2)
    orderM1HalfInner = <int> (md.order/2)
    md.coeffsSing = <double*> malloc(md.order*(pl.nData-1)*sizeof(double))
    md.coeffsNonsing = <double*> malloc(md.order*(pl.nData-1)*sizeof(double))
    if (NULL == md.coeffsSing or NULL == md.coeffsNonsing):
        destroy_fat_trapezoidalEndCorr(pl)
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
                destroy_fat_trapezoidalEndCorr(pl)
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
                destroy_fat_trapezoidalEndCorr(pl)
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
                destroy_fat_trapezoidalEndCorr(pl)
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
        destroy_fat_trapezoidalEndCorr(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # Input modification filter
    if pl.forwardBackward == 1:
        md.orderFilter = md.order+1 + (md.order % 2)
        md.coeffsFilter = <double*> malloc(md.orderFilter*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_trapezoidalEndCorr(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        with gil:
            try:
                coeffs_filter_memView = numpy.load(os.path.dirname(__file__) + "/data/coeffs_deriv_smooth_" + "%02d" % (md.orderFilter-1) + ".npy")
            except:
                destroy_fat_trapezoidalEndCorr(pl)
                raise
        for ii in range(md.orderFilter):
            md.coeffsFilter[ii] = coeffs_filter_memView[ii]*(-constants.piinv)
    elif pl.forwardBackward == 2 or pl.forwardBackward == -1 or pl.forwardBackward == -2:
        md.orderFilter = 1
        md.coeffsFilter = <double*> malloc(1*sizeof(double))
        if NULL == md.coeffsFilter:
            destroy_fat_trapezoidalEndCorr(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        if pl.forwardBackward == 2:            
            md.coeffsFilter[0] = -constants.piinv*pl.stepSize
        else:
            md.coeffsFilter[0] = 2.*pl.stepSize
    else:
        destroy_fat_trapezoidalEndCorr(pl)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    return 0


# Execute end-corrected trapezoidal
cdef int execute_fat_trapezoidalEndCorr(abel_plan* pl, double* dataIn, double* dataOut, int leftBoundary, int rightBoundary) nogil except -1:

    cdef:
        int ii, jj, nn
        methodData_EndCorr* md
        double* dataInTemp0 = NULL
        double* dataInTemp1 = NULL
        int orderM1Half, orderFilterM1Half, orderM1HalfInner
        int nLeftExt, nRightExt

    md = <methodData_EndCorr*> pl.methodData
    orderM1Half = <int> ((md.order-1)/2)
    orderM1HalfInner = <int> (md.order/2)
    orderFilterM1Half = (md.orderFilter-1)/2

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
    if rightBoundary == 0: # TODO or rightBoundary == 1 or rightBoundary == 2:
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
    
    # Main trapezoidal rule
    memset(dataOut, 0, pl.nData*sizeof(double))
    if pl.forwardBackward == -1:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData):
                dataOut[ii] += dataInTemp1[orderM1Half+jj]*pl.grid[jj]/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
    elif pl.forwardBackward == 1 or pl.forwardBackward == 2:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData):
                dataOut[ii] += dataInTemp1[orderM1Half+jj]/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
    elif pl.forwardBackward == -2:
        for ii in range(pl.nData-1):
            for jj in range(ii+1, pl.nData):
                dataOut[ii] += dataInTemp1[orderM1Half+jj]*(pl.grid[ii]/pl.grid[jj])**2/mathFun.sqrt(pl.grid[jj]**2-pl.grid[ii]**2)
    else:
        free(dataInTemp1)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # End correction right fairly smooth end
    for ii in range(pl.nData-1):
        for nn in range(md.order):
            jj = pl.nData-1+nn-orderM1HalfInner+orderM1Half
            dataOut[ii] += md.coeffsNonsing[md.order*ii+nn]*dataInTemp1[jj]

    # End correction left singular end
    for ii in range(pl.nData-1):
        for nn in range(md.order):
            dataOut[ii] += md.coeffsSing[md.order*ii+nn]*dataInTemp1[ii+nn]

    free(dataInTemp1)

    return 0


cdef int destroy_fat_trapezoidalEndCorr(abel_plan* pl) nogil except -1:

    cdef:
        methodData_EndCorr* md

    md = <methodData_EndCorr*> pl.methodData
    free(md.coeffsSing)
    free(md.coeffsNonsing)
    free(md.coeffsFilter)
    free(md)

    return 0

	




############################################################################################################################################


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




############################################################################################################################################








