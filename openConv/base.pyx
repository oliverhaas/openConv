

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from openConv.trap cimport plan_conv_trap, execute_conv_trap, destroy_conv_trap
from openConv.fft cimport plan_conv_fft, execute_conv_fft, destroy_conv_fft
from openConv.fmm cimport plan_conv_fmmCheb, execute_conv_fmmCheb, destroy_conv_fmmCheb, \
                          plan_conv_fmmExpCheb, execute_conv_fmmExpCheb, destroy_conv_fmmExpCheb
cimport openConv.interpolate as interp
import openConv.coeffs as coeffs

    
############################################################################################################################################
### Convolution of 1D symmetric smooth kernels                                                                                           ###
############################################################################################################################################


# Create plan for convolution
cdef conv_plan* plan_conv(int nData, int symData, double* kernel, int nKernel, int symKernel, double stepSize, int nDataOut,
                          funPtr kernelFun = NULL, void* kernelFunPar = NULL, double shiftData = 0., double shiftKernel = 0., 
                          int leftBoundaryKernel = 0, int rightBoundaryKernel = 0, int method = 0, int order = 3, 
                          double eps = 1.e-15) nogil except NULL:

    cdef:
        conv_plan* pl
        int ii
        int orderM1Half, orderM1HalfInner
        double xx
        double[::1] coeffs_smooth_memView

    # Small data set
    if nData < order+2:
        with gil:
            raise ValueError('Not enough data points for given parameters.')
    
    # For now restrict symmetries already here; will be extended in the future.
    if not (symData == 1 or symData == 2) or not (symKernel == 1 or symKernel == 2):
        with gil:
            raise NotImplementedError('Not implemented for given parameters.')
            
    # Set up plan struct
    pl = <conv_plan*> malloc(sizeof(conv_plan))
    if NULL == pl:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        pl.methodData = NULL
        pl.kernel = NULL
        pl.coeffsSmooth = NULL
    pl.method = method
    pl.order = order
    orderM1Half = <int> ((pl.order-1)/2)
    orderM1HalfInner = <int> (pl.order/2)
    pl.nData = nData
    pl.shiftData = shiftData
    pl.symData = symData
    pl.nKernel = nKernel
    pl.shiftKernel = shiftKernel
    pl.symKernel = symKernel
    pl.nDataOut = nDataOut
    pl.stepSize = stepSize

    # Check for correct kernel size
    if pl.nKernel < pl.nDataOut+pl.nData-1:  
        if NULL == kernelFun:
            with gil:
                raise ValueError('Not enough kernel data points for given parameters.')
        else:
            pl.nKernel = pl.nDataOut+pl.nData-1
            pl.kernel = <double*> malloc((nKernel+2*orderM1Half)*sizeof(double))
            for ii in range(nKernel+2*orderM1Half):
                xx = (ii+pl.shiftKernel-orderM1Half)*pl.stepSize
                kernelFun(&xx, kernelFunPar, &pl.kernel[ii])
            leftBoundaryKernel = 3
            rightBoundaryKernel = 3
    else:
        pl.kernel = cpExt(kernel, pl.nKernel, leftBoundaryKernel, pl.shiftKernel, rightBoundaryKernel, 0., pl.order)

    with gil:
        try:
            if pl.method == 0:
                plan_conv_trap(pl)
            elif pl.method == 1:
                plan_conv_fft(pl)
            elif pl.method == 2:
                plan_conv_fmmCheb(pl, kernelFun = kernelFun, kernelFunPar = kernelFunPar, eps = eps)
            elif pl.method == 3:
                plan_conv_fmmExpCheb(pl, kernelFun = kernelFun, kernelFunPar = kernelFunPar, eps = eps)
            else:
                with gil:
                    raise NotImplementedError('Method not implemented for given parameters.')
        except:
            # TODO
            destroy_conv_trap(pl)
            raise

    # Load and prepare end correction coefficients
    if pl.order > 0:    # Now zero order should work as well
        pl.coeffsSmooth = <double*> malloc(pl.order*sizeof(double))
        if NULL == pl.coeffsSmooth:
            destroy_conv_trap(pl)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        with gil:
            try:
                if pl.shiftKernel == 0.:
                    coeffs_smooth_memView = coeffs.getCoeffs('coeffs_smooth', pl.order)
                elif pl.shiftKernel == 0.5:
                    coeffs_smooth_memView = coeffs.getCoeffs('coeffs_smooth_halfShift', pl.order)
                else:
                    destroy_conv_trap(pl)
                    raise ValueError('Shift not allowed.')
            except:
                destroy_conv_trap(pl)
                raise
        for ii in range(pl.order):
            pl.coeffsSmooth[ii] = coeffs_smooth_memView[ii]
    
    return pl


# Execute given plan for Abel transform
cdef int execute_conv(conv_plan* pl, double* dataIn, double* dataOut, int leftBoundary = 0, int rightBoundary = 0) nogil except -1:

    cdef:
        double* dataInCpExt
        int ii
        
    if NULL == pl or NULL == dataIn or NULL == dataOut:
        with gil:
            raise TypeError('Input parameter is NULL.')

    # Copy input and extend/extrapolate if necessary
    dataInCpExt = cpExt(dataIn, pl.nData, leftBoundary, pl.shiftKernel, rightBoundary, 0., pl.order)

    # Set output to zero first
    memset(dataOut, 0, pl.nDataOut*sizeof(double))

    if pl.method == 0:
        execute_conv_trap(pl, dataInCpExt, dataOut)
    elif pl.method == 1:
        execute_conv_fft(pl, dataInCpExt, dataOut)
    elif pl.method == 2:
        execute_conv_fmmCheb(pl, dataInCpExt, dataOut)
    elif pl.method == 3:
        execute_conv_fmmExpCheb(pl, dataInCpExt, dataOut)
    else:
        free(dataInCpExt)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

#    # Do end corrections
#    endCorrections(pl, dataInCpExt, dataOut)
    
    # Free temporary dataIn array
    free(dataInCpExt)
    
    # Multiply by stepSize
    for ii in range(pl.nDataOut):
        dataOut[ii] *= pl.stepSize

    # Manually enforce some values due to symmetry
    if (pl.symData == 1 and pl.symKernel == 2) or (pl.symData == 2 and pl.symKernel == 1):
        dataOut[0] = 0.

    return 0

# Destroy given plan for Abel transform
cdef int destroy_conv(conv_plan* pl) nogil except -1:

    if not NULL == pl:
        if pl.method == 0:
            destroy_conv_trap(pl)
        elif pl.method == 1:
            destroy_conv_fft(pl)
        elif pl.method == 2:
            destroy_conv_fmmCheb(pl)
        elif pl.method == 3:
            destroy_conv_fmmExpCheb(pl)
        else:
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')

        free(pl.coeffsSmooth)
        free(pl.kernel)
        free(pl)    

    return 0



############################################################################################################################################


cdef double* cpExt(double* dataIn, int nData, int lBnd, double shL, int rBnd, double shR, int order) nogil except NULL:

    cdef:
        double* dataOut
        int ii, nLExt, nRExt
        int orderM1Half = <int> ((order-1)/2)
        
    if lBnd != 3:
        nLExt = orderM1Half
    else:
        nLExt = 0
    if rBnd != 3:
        nRExt = orderM1Half
    else:
        nRExt = 0
    
    dataOut = <double*> malloc((nData+order-1)*sizeof(double))
    if NULL == dataOut:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    for ii in range(nData+order-1-nLExt-nRExt):
        dataOut[nLExt+ii] = dataIn[ii]

    # Copy and extend data as specified
    if lBnd == 0:   # no symmetry, just extrapolation
        for ii in range(nLExt):
            dataOut[ii] = interp.polInt(&dataOut[nLExt], order, ii-nLExt)
    elif lBnd == 1: # odd symmetry
        if shL == 0.:
            for ii in range(nLExt):
                dataOut[nLExt-1-ii] = -dataOut[nLExt+1+ii]
        elif shL == 0.5:
            for ii in range(nLExt):
                dataOut[nLExt-1-ii] = -dataOut[nLExt+ii]
        else:
            free(dataOut)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif lBnd == 2: # even symmetry
        if shL == 0.:
            for ii in range(nLExt):
                dataOut[nLExt-1-ii] = dataOut[nLExt+1+ii]
        elif shL == 0.5:
            for ii in range(nLExt):
                dataOut[nLExt-1-ii] = dataOut[nLExt+ii]
        else:
            free(dataOut)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif lBnd == 3:
        pass
    else:
        free(dataOut)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
    if rBnd == 0:   # no symmetry, just extrapolation
        for ii in range(nRExt):
            dataOut[nData+nLExt+ii] = interp.polInt(&dataOut[nData+nLExt-order], order, ii+order)
    elif rBnd == 1: # odd symmetry
        if shR == 0.:
            for ii in range(nRExt):
                dataOut[nData+nLExt+ii] = -dataOut[nData+nLExt-2-ii]
        elif shR == 0.5:
            for ii in range(nRExt):
                dataOut[nData+nLExt+ii] = -dataOut[nData+nLExt-1-ii]
        else:
            free(dataOut)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif rBnd == 2: # even symmetry
        if shR == 0.:
            for ii in range(nLExt):
                dataOut[nData+nLExt+ii] = dataOut[nData+nLExt-2-ii]
        elif shR == 0.5:
            for ii in range(nLExt):
                dataOut[nData+nLExt+ii] = dataOut[nData+nLExt-1-ii]
        else:
            free(dataOut)
            with gil:
                raise NotImplementedError('Method not implemented for given parameters.')
    elif rBnd == 3:
        pass
    else:
        free(dataOut)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
            
    
    return dataOut




cdef double symSignFac(int sym) nogil:

    if sym == 1:
        return -1.
    elif sym == 2:
        return 1.
    else:
        return 0.


############################################################################################################################################


# End corrections
cdef int endCorrections(conv_plan* pl, double* dataIn, double* dataOut) nogil except -1:

    cdef:
        int ii, jj
        int orderM1Half, orderM1HalfDiff
        double signData, signKernel

    if pl.order == 0:
        return 0
    elif pl.order < 0:
        with gil:
            raise ValueError('Negative orders not defined.')
    else:
        # Some helper numbers
        orderM1Half = <int> ((pl.order-1)/2)
        orderM1HalfDiff = (<int> (pl.order/2)) - orderM1Half
        
        # Symmetries
        signData = symSignFac(pl.symData)
        signKernel = symSignFac(pl.symKernel)
        
        # Do trapezoidal part, but only if not midpoint rule
        if pl.shiftKernel == 0.:
            for ii in range(pl.nData):    # Tail right
                dataOut[ii] += signKernel*(-0.5)*pl.kernel[pl.nData-ii-1+orderM1Half]*dataIn[pl.nData-1+orderM1Half]
            for ii in range(pl.nData,pl.nDataOut):  # Tail data right
                dataOut[ii] += (-0.5)*pl.kernel[ii-pl.nData+1+orderM1Half]*dataIn[pl.nData-1+orderM1Half]
            for ii in range(pl.nDataOut):   # Tail left axis left
                dataOut[ii] += signData*(-0.5)*pl.kernel[pl.nData+ii-1+orderM1Half]*dataIn[pl.nData-1+orderM1Half]        
        
        if pl.order > 1:
            # TODO check for midpoint/halfShift
            # Do more corrections
            for ii in range(pl.nData-1):
                for jj in range(pl.order): # Central kernel right
                    dataOut[ii] += signKernel*pl.coeffsSmooth[pl.order-1-jj]*pl.kernel[jj]*dataIn[ii+jj]
            for ii in range(pl.nData-1):
                for jj in range(pl.order): # Tail right
                    dataOut[ii] += signKernel*pl.coeffsSmooth[jj] * \
                                   pl.kernel[pl.nData-ii-1+jj-orderM1HalfDiff]*dataIn[pl.nData-1+jj-orderM1HalfDiff]
            for ii in range(pl.nData,pl.nDataOut):
                for jj in range(pl.order): # Tail data right
                    dataOut[ii] += pl.coeffsSmooth[jj]*pl.kernel[ii-pl.nData-jj+pl.order]*dataIn[pl.nData-1+jj-orderM1HalfDiff]
            for ii in range(1,pl.nData):
                for jj in range(pl.order): # Central kernel left
                    dataOut[ii] += pl.coeffsSmooth[jj]*pl.kernel[pl.order-1-jj]*dataIn[ii+jj-orderM1HalfDiff]
            for ii in range(1,pl.nDataOut):
                for jj in range(pl.order): # Axis right
                    dataOut[ii] += pl.coeffsSmooth[pl.order-1-jj]*pl.kernel[ii+pl.order-orderM1HalfDiff-1-jj]*dataIn[jj]
            for ii in range(pl.nDataOut):
                for jj in range(pl.order): # Axis left
                    dataOut[ii] += signData*pl.coeffsSmooth[jj]*pl.kernel[ii+pl.order-1-jj]*dataIn[pl.order-1-jj]
            for ii in range(pl.nDataOut):
                for jj in range(pl.order):    # Tail left axis left
                    dataOut[ii] += signData*pl.coeffsSmooth[pl.order-1-jj] * \
                                   pl.kernel[pl.nData+ii-2+pl.order-orderM1HalfDiff-jj]*dataIn[pl.nData-2+pl.order-orderM1HalfDiff-jj]

    return 0



    

