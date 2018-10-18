

cdef double sign(double xx) nogil:
    return (1. if xx >=0 else -1.)
