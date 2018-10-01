
cimport openConv.interpolate as interp


ctypedef struct conv_plan:
    int method
    void* methodData
    int nData
    double shiftData
    int (*kernelFun)(double*, void*, double*) nogil
    interp.Interpolator* interpolator


cdef conv_plan* plan_conv(int nData, double shiftData, int (*kernelFun)(double*, void*, double*) nogil, int nKernel, double shiftKernel, 
                          double stepSize, int leftBoundaryKernel = ?, int rightBoundaryKernel = ?, int method = ?, int order = ?,
                          double eps = ?) nogil except NULL
cdef int execute_conv(conv_plan* plan, double* dataIn, double* dataOut, int leftBoundary = ?, int rightBoundary = ?) nogil except -1
cdef int destroy_conv(conv_plan* plan) nogil except -1
