# Simple script to convert *.h5 files output by Mathematica to *.npy files, which are
# faster and easier to lead by Python/Numpy/Cython.

import h5py
import numpy
import os

allFileNames = os.listdir('.')



pyxCont = '\n\ncdef:\n'

for fileName in allFileNames:
    if not (os.path.splitext(fileName)[1] == '.h5' and 'coeffs_smooth_' in os.path.splitext(fileName)[0]):
        allFileNames.remove(fileName)

allFileNames = sorted(allFileNames)

for ii in range(len(allFileNames)):
    pyxCont = pyxCont + '    double* ' + os.path.splitext(allFileNames[ii])[0] + ' = ['
    ind = len(pyxCont.splitlines()[-1])
    file = h5py.File(allFileNames[ii], 'r')
    data = file.get(file.keys()[0]).value
    for jj in range(len(data)-1):
        pyxCont = pyxCont + ('%.16e' % data[jj]) + ','
        currentInd = len(pyxCont.splitlines()[-1])
        if currentInd > 100:
            pyxCont = pyxCont + '\n' + ind*' '
        else:
            pyxCont = pyxCont + ' '
    pyxCont = pyxCont + ('%.16e' % data[len(data)-1]) + ']\n'

pyxCont = pyxCont + '    double** coeffs_smooth = ['
ind = len(pyxCont.splitlines()[-1])
for ii in range(len(allFileNames)):
    pyxCont = pyxCont + os.path.splitext(allFileNames[ii])[0] + ','
    currentInd = len(pyxCont.splitlines()[-1])
    if currentInd > 100:
        pyxCont = pyxCont + '\n' + ind*' '
    else:
        pyxCont = pyxCont + ' '
pyxCont = pyxCont + os.path.splitext(allFileNames[ii])[0] + ']\n'


pyxCont = pyxCont + 2*'\n' + \
"""
cdef double* getCoeffsSmooth(int order) nogil except NULL:
    if (<int> (sizeof(coeffs_smooth) / sizeof(coeffs_smooth[0])) < order:
        with gil:
            raise ValueError('Input order not available.')
    else:
        return coeffs_smooth[order-1]

"""

pyxFile = open("coeffs.pyx","w")
pyxFile.write(pyxCont)
#        file = h5py.File(fileName, 'r')
#        data = file.get(file.keys()[0]).value
#        numpy.save(os.path.splitext(fileName)[0] + '.npy', data)
