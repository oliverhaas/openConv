


from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy


cimport openConv.mathFun as mathFun
cimport scipy.linalg.cython_lapack as lapack





#####################################################################################################################################
# Polynomial inter-/extrapolation on equidistant grid
cdef double polInt(double* data, int nData, double xx) nogil:

    cdef:
        int ii, mm, ns
        double* cc = NULL
        double* dd = NULL
        double den, res

    cc = <double*> malloc(nData*sizeof(double))
    dd = <double*> malloc(nData*sizeof(double))
    if NULL == cc or NULL == dd:
        free(cc)
        free(dd)
        with gil:
            raise MemoryError('Malloc ruturned a NULL pointer, probably not enough memory available.')
    ns = <int> (xx+0.5)
    ns = min(max(ns,0),nData-1)
    for ii in range(nData):
        cc[ii] = data[ii]
        dd[ii] = data[ii]

    res = data[ns]
    ns -= 1
    for mm in range(1,nData):
        for ii in range(nData-mm):
            den = (dd[ii]-cc[ii+1])/mm
            dd[ii] = (ii+mm-xx)*den
            cc[ii]= (ii-xx)*den
        if 2*(ns+1) < nData-mm:
            res += cc[ns+1]
        else:
            res += dd[ns]
            ns -= 1
    free(cc)
    free(dd)

    return res



####################################################################################################################################

#####################################################################################################################################

cdef:
    int one = 1
    int two = 2
    double zerod = 0.

ctypedef struct spline1D1DEquiHelper:
    double xMin, xMax, dx, dxi
    int nx, nc, degree
    double* cc

# Inspired by numerical recipes
cdef Interpolator* Spline1D1DEquiFromData(double* yy, double xMin, double xMax, int nx, int degree = 3, int nExtLeft = 0, int typeLeft = 0, 
                                          int nDersLeft = 0, int* derOrdsLeft = NULL, double* derValsLeft = NULL, int nExtRight = 0,
                                          int typeRight = 0, int nDersRight = 0, int* derOrdsRight = NULL, 
                                          double* derValsRight = NULL) nogil except NULL:

    cdef:
        Interpolator* interp
        spline1D1DEquiHelper* helper
        int ii, jj, kk, ll
        int nLeft, nRight, nFillLeft, nFillRight
        double* ab
        double* abBound
        int mBound, nBound
        double* bBound
        int kl, ku, ldab
        double* bSplineStencil
        int* ipiv
        double temp

    if degree > 0 and degree % 2 == 0:
        with gil:
            raise NotImplementedError('Only odd degree (and zero) supported for now.')

    # Is there too much or not enough additional information at the boundaries?
    if nDersLeft + nExtLeft > (degree-1)/2 or (nDersLeft + nExtLeft < (degree-1)/2 and typeLeft < 0):
        with gil:
            raise ValueError('Not right number of boundary information.')
    if nDersRight + nExtRight > (degree-1)/2 or (nDersRight + nExtRight < (degree-1)/2 and typeRight < 0):
        with gil:
            raise ValueError('Not right number of boundary information.')

    interp = <Interpolator*> malloc(sizeof(Interpolator))       # TODO null check
    helper = <spline1D1DEquiHelper*> malloc(sizeof(spline1D1DEquiHelper))       # TODO null check

    interp.helper = helper
    interp.interpolate = _interpolate_Spline1D1DEqui
    interp.interpolateD = _interpolateD_Spline1D1DEqui
    interp.free = _free_Spline1D1DEqui

    helper.xMin = xMin
    helper.xMax = xMax
    helper.nx = nx
    helper.degree = degree
    helper.dx = (xMax-xMin)/(helper.nx-1)
    helper.dxi = 1./helper.dx
    if degree == 0 or degree == 1:
        helper.cc = <double*> malloc(nx*sizeof(double))
        helper.nc = nx
        for ii in range(nx):
            helper.cc[ii] = yy[ii]
        return interp
    
    helper.nc = nx + (degree-1)
    helper.cc = <double*> malloc(helper.nc*sizeof(double))     # TODO null check

    kl = (degree-1)/2
    ku = degree - 1 - kl
    ldab = 2*kl+ku+1
    ab = <double*> malloc(ldab*helper.nc*sizeof(double))     # TODO null check
    for ii in range(ldab):
        for jj in range(helper.nc):
            ab[ldab*jj+ii] = 0.

    # Sorted like this: first the "explicitly" given boundary information
    # Then Derivates, then extended grid points
    bSplineStencil = <double*> malloc(degree*sizeof(double))     # TODO null check
    for ii in range(degree):
        bSplineStencil[ii] = bSplineEqui(ii+1., degree)
    for ii in range(nx):
        helper.cc[kl+ii] = yy[nExtLeft+ii]
    for ii in range(kl,ldab):
        for jj in range((degree-1)/2,helper.nc-(degree-1)/2):
            ab[ldab*jj+ii] = bSplineStencil[ii-kl]
    free(bSplineStencil)


    # Now let's really take care of the boundaries, in seperate arrays at first
    nFillLeft = (degree-1)/2 - nExtLeft - nDersLeft
    mBound = (degree-1)/2 + 2 + (typeLeft==1)*nFillLeft
    nBound = degree + 1 + (typeLeft==1)*nFillLeft
    abBound = <double*> calloc(mBound*nBound, sizeof(double))
    bBound = helper.cc
    # Implicitely given boundary information
    if typeLeft == -1:  # Only derivates and extended grid points given
        pass
    elif typeLeft == 0:   # Natural spline
        for ii in range(nFillLeft):
            for jj in range(degree):
                abBound[ii*nBound+jj] = bSplineEquiD(degree-jj, degree, degree-1-ii)#*helper.dx**(degree-1-ii)
            bBound[ii] = 0.
    elif typeLeft == 1:   # Not-a-knot spline
        for ii in range(nFillLeft):
            for jj in range(nBound):
                abBound[ii*nBound+jj] = bSplineEquiD(degree+0.5+ii-jj, degree, degree) - \
                                        bSplineEquiD(degree+1.5+ii-jj, degree, degree)            # TODO check
            bBound[ii] = 0.
    else:
        free(ab)
        free(bSplineStencil)
        free(abBound)
        with gil:
            raise NotImplementedError('This kind of boundaries not implemented.')
    # Explicitely given boundary derivatives   
    for ii in range(nDersLeft):
        for jj in range(degree):
            abBound[(ii+nFillLeft)*nBound+jj] = bSplineEquiD(degree-jj, degree, derOrdsLeft[ii])   
        bBound[nFillLeft+ii] = derValsLeft[ii]
    # Explicitely given extended grid points   
    for ii in range(nExtLeft):
        for jj in range(degree+1):
            abBound[(ii+nFillLeft+nDersLeft)*nBound+jj] = bSplineEqui(nExtLeft+1+ii-jj, degree, ex = 1, exInterv = degree-jj)
        bBound[nFillLeft+nDersLeft+ii] = yy[ii]
    kk = nFillLeft+nDersLeft+nExtLeft
    ll = mBound-kk
    for ii in range(ll):
        for jj in range(nBound):
            abBound[(ii+kk)*nBound+jj] = bSplineEqui(nBound-ll+1+ii-jj, degree)

    for ii in range(mBound-1,0,-1):
        for jj in range(ii):
            temp = abBound[jj*nBound+nBound-1-(mBound-1-ii)]/abBound[ii*nBound+nBound-1-(mBound-1-ii)]
            for kk in range(nBound-(mBound-1-ii)-1):
                abBound[jj*nBound+kk] -= temp*abBound[ii*nBound+kk]
            abBound[jj*nBound+nBound-1-(mBound-1-ii)] = 0.        # TODO check
            bBound[jj] -= temp*bBound[ii]

    for ii in range((degree-1)/2):
        ab[ldab*ii+ku+kl] = abBound[ii*nBound+ii]
    for jj in range(ku):
        for ii in range((degree-1)/2):
            ab[ldab*(ii+jj+1)+kl+ku-1-jj] = abBound[ii*nBound+(jj+ii+1)]
    for jj in range(kl):
        for ii in range((degree-1)/2):
            ab[ldab*ii+kl+ku+1+jj] = abBound[(ii+1+jj)*nBound+ii]
    free(abBound)

    nFillRight = (degree-1)/2 - nExtRight - nDersRight
    mBound = (degree-1)/2 + 2 + (typeRight==1)*nFillRight
    nBound = degree + 1 + (typeRight==1)*nFillRight
    abBound = <double*> calloc(mBound*nBound, sizeof(double))
    bBound = &helper.cc[helper.nc-mBound]
    # Implicitely given boundary information
    if typeRight == -1:  # Only derivates and extended grid points given
        pass
    elif typeRight == 0:   # Natural spline
        for ii in range(nFillRight):
            for jj in range(nBound-degree,nBound):
                abBound[(mBound-1-ii)*nBound+jj] = bSplineEquiD(nBound-jj, degree, degree-1-ii)#/helper.dx**(degree-1-ii)
            bBound[mBound-1-ii] = 0.
    elif typeRight == 1:   # Not-a-knot spline
        for ii in range(nFillRight):
            for jj in range(nBound):
                abBound[(mBound-1-ii)*nBound+jj] = bSplineEquiD(degree+nFillRight-0.5-ii-jj, degree, degree) - \
                                                   bSplineEquiD(degree+nFillRight+0.5-ii-jj, degree, degree)        # TODO check
            bBound[mBound-1-ii] = 0.
    else:
        free(ab)
        free(bSplineStencil)
        free(abBound)
        with gil:
            raise NotImplementedError('This kind of boundaries not implemented.')
    # Explicitely given boundary derivatives   
    for ii in range(nDersRight):
        for jj in range(nBound-degree,nBound):
            abBound[(mBound-1-ii-nFillRight)*nBound+jj] = bSplineEquiD(nBound-jj, degree, derOrdsRight[ii])   
        bBound[mBound-1-nFillRight-ii] = derValsRight[ii]
    # Explicitely given extended grid points   
    for ii in range(nExtRight):
        for jj in range(nBound-degree-1,nBound):
            abBound[(mBound-nExtRight-nFillRight-nDersRight+ii)*nBound+jj] = bSplineEqui(degree+2+ii-jj, degree, ex = 1, exInterv = degree-jj)    # TODO check (degree-1)/2
        bBound[mBound-1-nExtRight-nFillRight-nDersRight+ii] = yy[nExtLeft+nx+ii]
    # Remaining matrix rows needed for rearranging the boundary part of the equation system
    kk = nFillRight+nDersRight+nExtRight
    ll = mBound-kk
    for ii in range(ll):
        for jj in range(nBound):
            abBound[ii*nBound+jj] = bSplineEqui(1+jj-ii, degree)

    for ii in range(mBound-1):
        for jj in range(ii+1,mBound):
            temp = abBound[jj*nBound+ii]/abBound[ii*nBound+ii]
            abBound[jj*nBound+ii] = 0.                                # TODO check
            for kk in range(ii+1,nBound):
                abBound[jj*nBound+kk] -= temp*abBound[ii*nBound+kk]
            bBound[jj] -= temp*bBound[ii]

    for ii in range((degree-1)/2):
        ab[ldab*(helper.nc-1-ii)+ku+kl] = abBound[(mBound-1-ii)*nBound+(nBound-1-ii)]
    for jj in range(ku):
        for ii in range((degree-1)/2):
            ab[ldab*(helper.nc-1-ii)+kl+ku-1-jj] = abBound[(mBound-2-ii-jj)*nBound+(nBound-1-ii)]
    for jj in range(kl):
        for ii in range((degree-1)/2):
            ab[ldab*(helper.nc-2-ii-jj)+kl+ku+1+jj] = abBound[(mBound-1-ii)*nBound+(nBound-2-ii-jj)]
    free(abBound)

    ipiv = <int*> malloc(helper.nc*sizeof(int))     # TODO null check
    lapack.dgbsv(&helper.nc, &kl, &ku, &one, ab, &ldab, ipiv, helper.cc, &helper.nc, &ii)

    free(ab)
    free(ipiv)

    return interp

# Inspired by the scipy/fitpack implementation
cdef int _interpolate_Spline1D1DEqui(Interpolator* me, double* xx, double* out) nogil:

    cdef:
        spline1D1DEquiHelper* helper = <spline1D1DEquiHelper*> me.helper
        double* bsplVals = <double*> malloc(2*(helper.degree+1)*sizeof(double))
        double* hh = bsplVals + helper.degree + 1
        double* h = bsplVals
        int ind, jj, nn, ell
        double xred, ww

    xred = (xx[0]-helper.xMin)*helper.dxi
    ell = min(max(<int> xred,0), helper.nc-helper.degree-1)

    # Normal de Boor iteration
    bsplVals[0] = 1.
    for jj in range(1,helper.degree+1):
        memcpy(hh, h, jj*sizeof(double))
        h[0] = 0.
        for nn in range(1,jj+1):
            ind = ell + nn
            ww = hh[nn-1]/jj
            h[nn-1] += ww*(ind-xred)
            h[nn] = ww*(xred-ind+jj)

    out[0] = 0.    
    for jj in range(helper.degree+1):
        out[0] += helper.cc[ell+jj]*bsplVals[jj]

    free(bsplVals)

    return 0

    

cdef int _interpolateD_Spline1D1DEqui(Interpolator* me, double* xx, double* out, int deriv) nogil:

    cdef:
        spline1D1DEquiHelper* helper = <spline1D1DEquiHelper*> me.helper
        double* bsplVals = <double*> malloc(2*(helper.degree+1)*sizeof(double))
        double* hh = bsplVals + helper.degree + 1
        double* h = bsplVals
        int ind, jj, nn, ell
        double xred, ww

    xred = (xx[0]-helper.xMin)*helper.dxi
    ell = min(max(<int> xred,0), helper.nc-helper.degree-1)

    # Normal de Boor iteration
    bsplVals[0] = 1.
    for jj in range(1,helper.degree-deriv+1):
        memcpy(hh, h, jj*sizeof(double))
        h[0] = 0.
        for nn in range(1,jj+1):
            ind = ell + nn
            ww = hh[nn-1]/jj
            h[nn-1] += ww*(ind-xred)
            h[nn] = ww*(xred-ind+jj)
    # Derivative iteration
    for jj in range(helper.degree-deriv+1,helper.degree+1):
        memcpy(hh, h, jj*sizeof(double))
        h[0] = 0.
        for nn in range(1,jj+1):
            ind = ell + nn
            h[nn-1] -= hh[nn-1]*helper.dxi
            h[nn] = hh[nn-1]*helper.dxi

    out[0] = 0.    
    for jj in range(helper.degree+1):
        out[0] += helper.cc[ell+jj]*bsplVals[jj]

    free(bsplVals)

    return 0


cdef void _free_Spline1D1DEqui(Interpolator* me) nogil:

    cdef:
        spline1D1DEquiHelper* helper = <spline1D1DEquiHelper*> me.helper

    free(helper.cc)
    free(helper)
    free(me)

    return



cdef double bSplineEqui(double xx, int degree, int ex = 0, int exInterv = 0) nogil:

    if degree < 0:
        with gil:
            raise ValueError('Degree smaller than 0 not defined.')
    elif degree == 0:
        if ex == 1 and exInterv == 0:
            return 1.
        elif ex == 0 and xx >= 0 and xx < 1.:
            return 1.
        else:
            return 0.
    else:
        return xx/degree*bSplineEqui(xx, degree-1, ex=ex, exInterv=exInterv) + \
               (degree+1-xx)/degree*bSplineEqui(xx-1., degree-1, ex=ex, exInterv=exInterv-1)


cdef double bSplineEquiD(double xx, int degree, int deriv, int ex = 0, int exInterv = 0) nogil:

    if degree < 0 or deriv < 0:
        with gil:
            raise ValueError('Degree and/or derivative smaller than 0 not defined.')
    elif deriv == 0:
        return bSplineEqui(xx, degree, ex=ex, exInterv = exInterv)
    else:
        return bSplineEquiD(xx, degree-1, deriv-1, ex=ex, exInterv=exInterv) - \
               bSplineEquiD(xx-1., degree-1, deriv-1, ex=ex, exInterv=exInterv-1)



cdef double bStepSplineEqui(double xx, int degree) nogil:

    if degree < 0:
        with gil:
            raise ValueError('Degree smaller than 0 not defined.')
    elif degree == 0:
        if xx >= 0.:
            return 1.
        else:
            return 0.
    else:
        return xx/degree*bStepSplineEqui(xx, degree-1) + \
               (degree-xx)/degree*bStepSplineEqui(xx-1., degree-1)



############################################################################################################################################

ctypedef struct newton1D1DEquiHelper:
    double xMin, xMax, dx, dxi
    int nx, degree
    double* data


cdef Interpolator* Newton1D1DEquiFromData(double* yy, double xMin, double xMax, int nx, int degree = 3) nogil except NULL:

    cdef:
        Interpolator* interp
        newton1D1DEquiHelper* helper
        int ii

    interp = <Interpolator*> malloc(sizeof(Interpolator))       # TODO null check
    helper = <newton1D1DEquiHelper*> malloc(sizeof(newton1D1DEquiHelper))       # TODO null check

    interp.helper = helper
    interp.interpolate = _interpolate_Newton1D1DEqui
    interp.interpolateD = _interpolateD_Newton1D1DEqui
    interp.free = _free_Newton1D1DEqui

    helper.xMin = xMin
    helper.xMax = xMax
    helper.nx = nx
    helper.degree = degree
    helper.dx = (xMax-xMin)/(helper.nx-1)
    helper.dxi = 1./helper.dx
    
    helper.nx = nx
    helper.data = <double*> malloc(helper.nx*sizeof(double))     # TODO null check

    for ii in range(nx):
        helper.data[ii] = yy[ii]

    return interp


# Inspired by the scipy/fitpack implementation
cdef int _interpolate_Newton1D1DEqui(Interpolator* me, double* xx, double* out) nogil:

    cdef:
        newton1D1DEquiHelper* helper = <newton1D1DEquiHelper*> me.helper
        int ind, ii, jj
        double xred
        double* cc = <double*> malloc((helper.degree+1)*sizeof(double))

    xred = (xx[0]-helper.xMin)*helper.dxi
    ind = min(max((<int> xred - (helper.degree-1)/2),0), helper.nx-helper.degree-1)

    for ii in range(helper.degree+1):
        cc[ii] = helper.data[ind+ii]

    xred = xred - ind

    for jj in range(1,helper.degree+1):
        for ii in range(helper.degree+1-jj):
            cc[ii] = (cc[ii+1]-cc[ii])/jj

    out[0] = cc[0]
    for ii in range(1,helper.degree+1):
        out[0] = out[0]*(xred-ii)+cc[ii]

    free(cc)

    return 0

    

cdef int _interpolateD_Newton1D1DEqui(Interpolator* me, double* xx, double* out, int deriv) nogil:

    cdef:
        newton1D1DEquiHelper* helper = <newton1D1DEquiHelper*> me.helper
        int ind, ii, jj
        double xred, temp
        double* cc = <double*> malloc((helper.degree+1)*sizeof(double))
        double* hh = <double*> malloc((deriv+1)*sizeof(double))

    xred = (xx[0]-helper.xMin)*helper.dxi
    ind = min(max((<int> xred - (helper.degree-1)/2),0), helper.nx-helper.degree-1)

    for ii in range(helper.degree+1):
        cc[ii] = helper.data[ind+ii]

    xred = xred - ind

    for jj in range(1,helper.degree+1):
        for ii in range(helper.degree+1-jj):
            cc[ii] = (cc[ii+1]-cc[ii])/jj

    hh[0] = cc[0]
    for jj in range(1,deriv+1):
        hh[jj] = 0.
    for ii in range(1, helper.degree+1):
        temp = (xred-ii)
        for jj in range(deriv+1,0,-1):
            hh[jj] = hh[jj]*temp + hh[jj-1]
        hh[0] = hh[0]*temp + cc[ii]
        
    out[0] = hh[deriv]*helper.dxi**deriv

    free(cc)
    free(hh)

    return 0


cdef void _free_Newton1D1DEqui(Interpolator* me) nogil:

    cdef:
        newton1D1DEquiHelper* helper = <newton1D1DEquiHelper*> me.helper

    free(helper.data)
    free(helper)
    free(me)

    return

















