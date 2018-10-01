

from libc.stdlib cimport malloc, free

#from openConv.fft cimport plan_conv_fftEndCorr, execute_conv_fftEndCorr, destroy_conv_fftEndCorr
from openConv.trap cimport plan_conv_trap, execute_conv_trap, destroy_conv_trap
#from openConv.fmm cimport plan_conv_fmmTrapEndCorr, execute_conv_fmmTrapEndCorr, destroy_conv_fmmTrapEndCorr
cimport openConv.interpolate as interp


ctypedef struct conv_plan:
    int method
    void* methodData
    int nData
    double shiftData
    int (*kernelFun)(double*, void*, double*) nogil
    interp.Interpolator* interpolator
#    int nKernel
#    double shiftKernel
#    int symKernel
#    double stepSize
#    int method
#    double* grid
#    void* methodData



############################################################################################################################################
### Convolution of 1D symmetric smooth kernels                                                                                           ###
############################################################################################################################################


#cdef conv_plan* plan_conv_fromData(int nData, double shiftData, double* kernel, int nKernel, double shiftKernel, double stepSize, 
#                          int leftBoundaryKernel = 0, int rightBoundaryKernel = 0, int method = 3, int order = 2, 
#                          double eps = 1.e-15) nogil except NULL:


# Create plan for convolution
cdef conv_plan* plan_conv(int nData, double shiftData, int (*kernelFun)(double*, void*, double*) nogil, int nKernel, double shiftKernel, double stepSize, 
                          int leftBoundaryKernel = 0, int rightBoundaryKernel = 0, int method = 3, int order = 2, 
                          double eps = 1.e-15) nogil except NULL:

    cdef:
        conv_plan* pl
        int ii

    pl = <conv_plan*> malloc(sizeof(conv_plan))
    if NULL == pl:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
#        pl.grid = NULL
        pl.methodData = NULL
        pl.interpolator = NULL
    pl.nData = nData
    # TODO
#    pl.forwardBackward = forwardBackward
#    pl.shift = shift
#    pl.stepSize = stepSize
#    pl.method = method
#    if NULL == pl.grid:
#        free(pl)
#        with gil:
#            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
#    for ii in range(nData):
#        pl.grid[ii] = (ii+shift)*stepSize

    with gil:
        try:
            if pl.method == 0:
                plan_conv_trap(pl, order = order)
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
#            free(pl.grid)
            free(pl)
            raise

    return pl


# Execute given plan for Abel transform
cdef int execute_conv(conv_plan* pl, double* dataIn, double* dataOut, int leftBoundary = 0, int rightBoundary = 0) nogil except -1:

    if NULL == pl:
        with gil:
            raise TypeError('Input plan is NULL.')

    if pl.method == 0:
        execute_conv_trap(pl, dataIn, dataOut, leftBoundary, rightBoundary)
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
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')


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

    # TODO
#    free(pl.grid)
    free(pl)    

    return 0






cdef double* cpExt(double* dataIn, int nData, int nLExt, int lBnd, double shL, 
                   int nRExt, int rBnd, double shR, int order) nogil except NULL:

    cdef:
        double* dataOut = <double*> malloc((nData+nLExt+nRExt)*sizeof(double))
        int ii

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
    else:
        free(dataOut)
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')
            
    
    return dataOut

















    

