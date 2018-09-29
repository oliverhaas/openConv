

from libc.stdlib cimport malloc, free

from openConv.fft cimport plan_conv_fftEndCorr, execute_conv_fftEndCorr, destroy_conv_fftEndCorr
from openConv.quad cimport plan_conv_trapEndCorr, execute_conv_trapEndCorr, destroy_conv_trapEndCorr
from openConv.fmm cimport plan_conv_fmmTrapEndCorr, execute_conv_fmmTrapEndCorr, destroy_conv_fmmTrapEndCorr



ctypedef struct conv_plan:
    int nData
    int forwardBackward
    double shift
    double stepSize
    int method
    double* grid
    void* methodData



############################################################################################################################################
### Fast Abel transforms                                                                                                                 ###
############################################################################################################################################


# Create plan for Abel transform
cdef conv_plan* plan_conv(int nData, int forwardBackward, double shift, double stepSize, 
                         int method = 3, int order = 2, double eps = 1.e-15) nogil except NULL:

    cdef:
        conv_plan* pl
        int ii

    pl = <conv_plan*> malloc(sizeof(conv_plan))
    if NULL == pl:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        pl.grid = NULL
        pl.methodData = NULL
    pl.nData = nData
    pl.forwardBackward = forwardBackward
    pl.shift = shift
    pl.stepSize = stepSize
    pl.method = method
    pl.grid = <double*> malloc(nData*sizeof(double))
    if NULL == pl.grid:
        free(pl)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    for ii in range(nData):
        pl.grid[ii] = (ii+shift)*stepSize

    with gil:
        try:
            if pl.method == 0:
                plan_conv_trapezoidalDesingConst(pl)
            elif pl.method == 1:
                plan_conv_hansenLawOrgLin(pl)
            elif pl.method == 2:
                plan_conv_trapezoidalEndCorr(pl, order = order)
            elif pl.method == 3:
                plan_conv_fmmTrapEndCorr(pl, order = order, eps = eps)
            else:
                with gil:
                    raise NotImplementedError('Method not implemented for given parameters.')
        except:
            free(pl.grid)
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
    elif pl.method == 1:
        execute_conv_fft(pl, dataIn, dataOut)
    elif pl.method == 2:
        execute_conv_fmmCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
    elif pl.method == 2:
        execute_conv_fmmExpCheb(pl, dataIn, dataOut, leftBoundary, rightBoundary)
    else:
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')


# Destroy given plan for Abel transform
cdef int destroy_conv(conv_plan* pl) nogil except -1:

    if NULL == pl:
        with gil:
            raise TypeError('Input plan is NULL.')

    if pl.method == 0:
        destroy_conv_trapezoidalDesingConst(pl)
    elif pl.method == 1:
        destroy_conv_hansenLawLinear(pl)
    elif pl.method == 2:
        destroy_conv_trapezoidalEndCorr(pl)
    elif pl.method == 3:
        destroy_conv_fmmTrapEndCorr(pl)
    else:
        with gil:
            raise NotImplementedError('Method not implemented for given parameters.')

    free(pl.grid)
    free(pl)    

    return 0



