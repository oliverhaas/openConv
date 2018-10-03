
import numpy as np
import datetime as dt


from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset, memcpy
cimport scipy.linalg.cython_blas as blas


cimport openConv.mathFun as math
cimport openConv.constants as const
cimport openConv.base as base


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
    void fftw_execute_r2r(const fftw_plan plan, double* inData, double* outData) nogil

    # Destroy plan
    void fftw_destroy_plan(fftw_plan plan) nogil

    # Print plan
    void fftw_print_plan(fftw_plan plan) nogil

# Direction enum standard fft
cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1

# Documented flags
cdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64
    FFTW_WISDOM_ONLY = 2097152

# Transform type real to real fft
cdef enum:
     FFTW_R2HC = 0
     FFTW_HC2R = 1
     FFTW_DHT = 2
     FFTW_REDFT00 = 3
     FFTW_REDFT01 = 4
     FFTW_REDFT10 = 5
     FFTW_REDFT11 = 6
     FFTW_RODFT00 = 7
     FFTW_RODFT01 = 8
     FFTW_RODFT10 = 9
     FFTW_RODFT11 = 10




############################################################################################################################################
### Fast convolution based on end corrected fast fourier transform                                                                       ###
############################################################################################################################################

ctypedef struct methodData_fft:
    int nfft
    double norm
    fftw_plan fftData   
    fftw_plan fftKernel
    fftw_plan fftResult
    double* kernelfft

# Plan convolution with fftezoidal rule
cdef int plan_conv_fft(conv_plan* pl) nogil except -1:

    cdef:
        int ii, jj, orderM1Half
        methodData_fft* md
        double* tempVec = NULL
        int typeFFTData, typeFFTKernel, typeFFTResult

    # Input check
    if NULL == pl:
        with gil:
            raise ValueError('Illegal input argument; plan is NULL.')   

    # Main method struct
    md = <methodData_fft*> malloc(sizeof(methodData_fft))
    if NULL == md:
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    else:
        md.kernelfft = NULL
        md.fftData = NULL
        md.fftKernel = NULL
        md.fftResult = NULL
    pl.methodData = <void*> md

    # Helpers    
    orderM1Half = <int> ((pl.order-1)/2)

    if (pl.symData != 1 and pl.symData != 2) or (pl.symKernel != 1 and pl.symKernel != 2):
        with gil:
            raise NotImplementedError('Symmetry not implemented (yet).')

    md.nfft = pl.nData+pl.nKernel-1
    tempVec = <double*> fftw_malloc(md.nfft*sizeof(double))
    md.kernelfft = <double*> fftw_malloc(md.nfft*sizeof(double))
    if NULL == tempVec or NULL == md.kernelfft:
        destroy_conv_fft(pl)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.') 

    # Treat different symmetries
    # Remark: The most efficient way for linear convolution seems to avoid using DCT-I,
    # due to it being the slowest and the one with different normalization.
    # I don't see any reason otherwise.
    # TODO maybe there is a way to make this easier?
    if pl.symData == 1 and pl.symKernel == 1:
        md.norm = -1./(2.*md.nfft)
    elif (pl.symData == 1 or pl.symData == 2) and (pl.symKernel == 1 or pl.symKernel == 2):
        md.norm = 1./(2.*md.nfft)
    else:
        destroy_conv_fft(pl)
        with gil:
            raise NotImplementedError('Symmetry not implemented (yet).')
    if (pl.shiftData == 0. and pl.shiftKernel == 0.) and (pl.symData == 2 and pl.symKernel == 2):
        typeFFTData = FFTW_REDFT01
        typeFFTKernel = FFTW_REDFT01
        typeFFTResult = FFTW_REDFT10
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.) and (pl.symData == 1 and pl.symKernel == 2):
        typeFFTData = FFTW_RODFT01
        typeFFTKernel = FFTW_REDFT01
        typeFFTResult = FFTW_RODFT10
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.) and (pl.symData == 2 and pl.symKernel == 1):
        typeFFTData = FFTW_REDFT01
        typeFFTKernel = FFTW_RODFT01
        typeFFTResult = FFTW_RODFT10
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.) and (pl.symData == 1 and pl.symKernel == 1):
        typeFFTData = FFTW_RODFT01
        typeFFTKernel = FFTW_RODFT01
        typeFFTResult = FFTW_REDFT10
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.5) and (pl.symData == 2 and pl.symKernel == 2):
        typeFFTData = FFTW_REDFT01
        typeFFTKernel = FFTW_REDFT11
        typeFFTResult = FFTW_REDFT11
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.5) and (pl.symData == 1 and pl.symKernel == 2):
        typeFFTData = FFTW_RODFT01
        typeFFTKernel = FFTW_REDFT11
        typeFFTResult = FFTW_RODFT11
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.5) and (pl.symData == 2 and pl.symKernel == 1):
        typeFFTData = FFTW_REDFT01
        typeFFTKernel = FFTW_RODFT11
        typeFFTResult = FFTW_RODFT10
    elif (pl.shiftData == 0. and pl.shiftKernel == 0.5) and (pl.symData == 1 and pl.symKernel == 1):
        typeFFTData = FFTW_RODFT01
        typeFFTKernel = FFTW_RODFT11
        typeFFTResult = FFTW_REDFT10
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.) and (pl.symData == 2 and pl.symKernel == 2):
        typeFFTData = FFTW_REDFT11
        typeFFTKernel = FFTW_REDFT01
        typeFFTResult = FFTW_REDFT11
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.) and (pl.symData == 1 and pl.symKernel == 2):
        typeFFTData = FFTW_RODFT11
        typeFFTKernel = FFTW_REDFT01
        typeFFTResult = FFTW_RODFT11
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.) and (pl.symData == 2 and pl.symKernel == 1):
        typeFFTData = FFTW_REDFT11
        typeFFTKernel = FFTW_RODFT01
        typeFFTResult = FFTW_RODFT10
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.) and (pl.symData == 1 and pl.symKernel == 1):
        typeFFTData = FFTW_RODFT11
        typeFFTKernel = FFTW_RODFT01
        typeFFTResult = FFTW_REDFT10
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.5) and (pl.symData == 2 and pl.symKernel == 2):
        typeFFTData = FFTW_REDFT10
        typeFFTKernel = FFTW_REDFT10
        typeFFTResult = FFTW_REDFT11
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.5) and (pl.symData == 1 and pl.symKernel == 2):
        typeFFTData = FFTW_RODFT10
        typeFFTKernel = FFTW_REDFT10
        typeFFTResult = FFTW_RODFT11
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.5) and (pl.symData == 2 and pl.symKernel == 1):
        typeFFTData = FFTW_REDFT10
        typeFFTKernel = FFTW_RODFT10
        typeFFTResult = FFTW_RODFT11
    elif (pl.shiftData == 0.5 and pl.shiftKernel == 0.5) and (pl.symData == 1 and pl.symKernel == 1):
        typeFFTData = FFTW_RODFT10
        typeFFTKernel = FFTW_RODFT10
        typeFFTResult = FFTW_REDFT11
    else:
        with gil:
            raise NotImplementedError('Symmetry or shift not implemented.')
            
    md.fftData = fftw_plan_r2r_1d(md.nfft, tempVec, md.kernelfft, typeFFTData, 64)
    md.fftKernel = fftw_plan_r2r_1d(md.nfft, tempVec, md.kernelfft, typeFFTKernel, 64)
    md.fftResult = fftw_plan_r2r_1d(md.nfft, tempVec, md.kernelfft, typeFFTResult, 64)
    
    # Do FFT of kernel already        
    memset(&md.kernelfft[pl.nKernel], 0, (md.nfft-pl.nKernel)*sizeof(double))
    memcpy(tempVec, &pl.kernel[orderM1Half], pl.nKernel*sizeof(double))
    fftw_execute(md.fftKernel)
    
    fftw_free(tempVec)

    
    return 0


# Execute convolution with fft
cdef int execute_conv_fft(conv_plan* pl, double* dataIn, double* dataOut) nogil except -1:

    cdef:
        methodData_fft* md
        int ii, jj, orderM1Half
        double* tempVec1
        double* tempVec2

    # Input check
    if NULL == pl or NULL == dataIn or NULL == dataOut:
        with gil:
            raise ValueError('Illegal input argument; something is NULL.')   
    
    # Get method data
    md = <methodData_fft*> pl.methodData
    
    # Helpers    
    orderM1Half = <int> ((pl.order-1)/2)

    if (pl.shiftData == 0. or pl.shiftData == 0.5) and (pl.symData == 1 and pl.symData == 2) and \
       (pl.shiftKernel == 0. or pl.shiftKernel == 0.5) and (pl.symKernel == 1 and pl.symKernel == 2):
        tempVec1 = <double*> fftw_malloc(md.nfft*sizeof(double)) 
        tempVec2 = <double*> fftw_malloc(md.nfft*sizeof(double)) 
        if NULL == tempVec1 or NULL == tempVec2:
            fftw_free(tempVec1)
            fftw_free(tempVec2)
            with gil:
                raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
        memset(&tempVec1[pl.nData], 0, (md.nfft-pl.nData)*sizeof(double))
        memcpy(tempVec1, &dataIn[orderM1Half], pl.nData*sizeof(double))
        fftw_execute_r2r(md.fftData, tempVec1, tempVec2)    # tempVec2 holds FFT of dataIn
        for ii in range(md.nfft):
            tempVec2[ii] *= md.kernelfft[ii]
        fftw_execute_r2r(md.fftResult, tempVec2, tempVec1)  # tempVec1 holds result (on extended grid)
        
        memcpy(dataOut, tempVec1, pl.nDataOut*sizeof(double))
        fftw_free(tempVec1)
        fftw_free(tempVec2)

        for ii in range(pl.nDataOut):
            dataOut[ii] *= md.norm
    else:
        with gil:
            raise NotImplementedError('Not implemented for given parameters.')    

    return 0


# Destroy convolution structs
cdef int destroy_conv_fft(conv_plan* pl) nogil except -1:

    cdef:
        methodData_fft* md = <methodData_fft*> pl.methodData

    fftw_free(md.kernelfft)
    if not NULL == md.fftData:
        fftw_destroy_plan(md.fftData)
    if not NULL == md.fftKernel:
        fftw_destroy_plan(md.fftKernel)
    if not NULL == md.fftResult:
        fftw_destroy_plan(md.fftResult)
    free(md)
    
    return 0

	




############################################################################################################################################




cdef int getTypeFFT(int sym, double shift) nogil:

    if sym == 1:
        pass
#     FFTW_RODFT00 = 7
#     FFTW_RODFT01 = 8
#     FFTW_RODFT10 = 9
#     FFTW_RODFT11 = 10        
    elif sym == 2:
        pass
#     FFTW_REDFT00 = 3
#     FFTW_REDFT01 = 4
#     FFTW_REDFT10 = 5
#     FFTW_REDFT11 = 6
    else:
        with gil:
            raise NotImplementedError('Symmetry not implemmented.')






