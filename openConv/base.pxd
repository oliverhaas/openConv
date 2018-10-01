
cimport openConv.interpolate as interp


ctypedef struct conv_plan:
    int method
    int order
    void* methodData
    int nData
    double shiftData
    int symData
    int nKernel
    double* kernel
    double shiftKernel
    int symKernel
    int (*kernelFun)(double*, void*, double*) nogil
    interp.Interpolator* interpKernel
    double stepSize
    double* coeffsSmooth

ctypedef int (*funPtr)(double*, void*, double*) nogil

cdef conv_plan* plan_conv(int nData, double shiftData, double* kernel, int nKernel, double shiftKernel, double stepSize,
                          funPtr kernelFun = ?, void* kernelFunPar = ?, int leftBoundaryKernel = ?,
                          int rightBoundaryKernel = ?, int method = ?, int order = ?, double eps = ?) nogil except NULL
cdef int execute_conv(conv_plan* plan, double* dataIn, double* dataOut, int leftBoundary = ?, int rightBoundary = ?) nogil except -1
cdef int destroy_conv(conv_plan* plan) nogil except -1






cdef double* cpExt(double* dataIn, int nData, int lBnd, double shL, int rBnd, double shR, int order) nogil except NULL
cdef double symSignFac(int sym) nogil
