



cdef unsigned int uintMax(unsigned int aa, unsigned int bb) nogil:

    if aa > bb:
        return aa

    return bb


cdef unsigned int uintMin(unsigned int aa, unsigned int bb) nogil:

    if aa < bb:
        return aa

    return bb




