

ctypedef struct abel_plan:
    int nData
    int forwardBackward
    double shift
    double stepSize
    int method
    double* grid
    void* methodData

cdef abel_plan* plan_fat(int nData, int forwardBackward, double shift, double stepSize, int method = ?, int order = ?, double eps = ?) nogil except NULL
cdef int execute_fat(abel_plan* plan, double* dataIn, double* dataOut, int leftBoundary = ?, int rightBoundary = ?) nogil except -1
cdef int destroy_fat(abel_plan* plan) nogil except -1
