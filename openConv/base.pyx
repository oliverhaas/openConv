

from libc.stdlib cimport malloc, free

#from openConv.fft cimport plan_conv_fftEndCorr, execute_conv_fftEndCorr, destroy_conv_fftEndCorr
from openConv.trap cimport plan_conv_trap, execute_conv_trap, destroy_conv_trap
#from openConv.fmm cimport plan_conv_fmmTrapEndCorr, execute_conv_fmmTrapEndCorr, destroy_conv_fmmTrapEndCorr
cimport openConv.interpolate as interp
cimport openConv.coeffs as coeffs

from openConv.base cimport conv_plan, funPtr    # import from own *.pxd 


    
############################################################################################################################################
### Convolution of 1D symmetric smooth kernels                                                                                           ###
############################################################################################################################################


#cdef conv_plan* plan_conv_fromData(int nData, double shiftData, double* kernel, int nKernel, double shiftKernel, double stepSize, 
#                          int leftBoundaryKernel = 0, int rightBoundaryKernel = 0, int method = 3, int order = 2, 
#                          double eps = 1.e-15) nogil except NULL:

# Create plan for convolution
cdef conv_plan* plan_conv(int nData, double shiftData, double* kernel, int nKernel, double shiftKernel, double stepSize,
                          funPtr kernelFun = NULL, void* kernelFunPar = NULL, int leftBoundaryKernel = 0,
                          int rightBoundaryKernel = 0, int method = 3, int order = 2, double eps = 1.e-15) nogil except NULL:

    cdef:
        conv_plan* pl
        int ii
        double* kernelTemp
        int orderM1Half, orderM1HalfInner
        double xx, val
        double[::1] coeffs_smooth_memView

    # Small data set
    if nData < order+2:
        with gil:
            raise ValueError('Not enough data points for given parameters.')
    
    # Set up plan struct
    pl = <conv_plan*> malloc(sizeof(conv_plan))
    if NULL == pl:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        pl.methodData = NULL
        pl.interpKernel = NULL
        pl.kernel = NULL
        pl.coeffsSmooth = NULL
    pl.method = method
    pl.order = order
    orderM1Half = <int> ((pl.order-1)/2)
    orderM1HalfInner = <int> (pl.order/2)
    pl.nData = nData
    pl.shiftData = shiftData
    pl.nKernel = nKernel
    pl.shiftKernel = shiftKernel
    
    # Check how kernel has been input
    if NULL == kernel:
        if NULL == kernelFun:
            with gil:
                destroy_conv(pl)
                raise TypeError('Either kernel data or kernel function pointer has to be not NULL.')
        else:
            pl.kernel = <double*> malloc((nKernel+2*orderM1Half)*sizeof(double))
            for ii in range(nKernel+2*orderM1Half):
                xx = (pl.shiftKernel+ii-orderM1Half)*pl.stepSize
                kernelFun(&xx, kernelFunPar, &pl.kernel[ii])
    else:
        pl.kernel = cpExt(kernel, nKernel, leftBoundaryKernel, pl.shiftKernel, rightBoundaryKernel, 0., pl.order)

    with gil:
        try:
            if pl.method == 0:
                plan_conv_trap(pl)
#            elif pl.method == 1:
#                plan_conv_fft(pl, order = order)
#            elif pl.method == 2:
#                plan_conv_fmmCheb(pl, order = order, eps = eps)
#            elif pl.method == 3:
#                plan_conv_fmmExpCheb(pl, order = order, eps = eps)
#            elif pl.method == 4:
#                plan_conv_fullCheb(pl, order = order, eps = eps)
#            elif pl.method == 5:
#                plan_conv_fullExpCheb(pl, order = order, eps = eps)
            else:
                with gil:
                    raise NotImplementedError('Method not implemented for given parameters.')
        except:
            # TODO
            destroy_conv_trap(pl)
            raise

    # Load and prepare end correction coefficients
    pl.coeffsSmooth = <double*> malloc(pl.order*sizeof(double))
    if NULL == pl.coeffsSmooth:
        destroy_conv_trap(pl)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    with gil:
        try:
            coeffs_smooth_memView = coeffs.getCoeffs('coeffs_smooth', pl.order)
        except:
            destroy_conv_trap(pl)
            raise
    for ii in range(pl.order):
        pl.coeffsSmooth[ii] = coeffs_smooth_memView[ii]
    pl.coeffsSmooth[orderM1HalfInner] += -0.5
    
    return pl


# Execute given plan for Abel transform
cdef int execute_conv(conv_plan* pl, double* dataIn, double* dataOut, int leftBoundary = 0, int rightBoundary = 0) nogil except -1:

    cdef:
        double* dataInCpExt

    if NULL == pl:
        with gil:
            raise TypeError('Input plan is NULL.')

    dataInCpExt = cpExt(dataIn, pl.nData, leftBoundary, pl.shiftKernel, rightBoundary, 0., pl.order)
        
    if pl.method == 0:
        execute_conv_trap(pl, dataInCpExt, dataOut)
#    elif pl.method == 1:
#        execute_conv_fft(pl, dataIn, dataOut, leftBoundary, rightBoundary)
#    elif pl.method == 2:
#        execute_conv_fmmCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
#    elif pl.method == 3:
#        execute_conv_fmmExpCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
#    elif pl.method == 4:
#        execute_conv_fullCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
#    elif pl.method == 5:
#        execute_conv_fullExpCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
    else:
        free(dataInCpExt)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
    free(dataInCpExt)
    
    return 0

# Destroy given plan for Abel transform
cdef int destroy_conv(conv_plan* pl) nogil except -1:

    if NULL == pl:
        with gil:
            raise TypeError('Input plan is NULL.')

    if pl.method == 0:
        destroy_conv_trap(pl)
#    elif pl.method == 1:
#        destroy_conv_fft(pl)
#    elif pl.method == 2:
#        destroy_conv_fmmCheb(pl)
#    elif pl.method == 3:
#        destroy_conv_fmmExpCheb(pl)
#    elif pl.method == 4:
#        destroy_conv_fullCheb(pl)
#    elif pl.method == 5:
#        destroy_conv_fullExpCheb(pl)
    else:
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    # TODO check
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
    
    dataOut = <double*> malloc((nData+nLExt+nRExt)*sizeof(double))
    if NULL == dataOut:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    for ii in range(nData):
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











    

