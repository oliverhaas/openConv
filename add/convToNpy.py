# Simple script to convert *.h5 files output by Mathematica to *.npy files, which are
# faster and easier to lead by Python/Numpy/Cython.

import h5py
import numpy
import os

allFileNames = os.listdir('.')

for fileName in allFileNames:
    if os.path.splitext(fileName)[1] == '.h5':
        file = h5py.File(fileName, 'r')
        data = file.get(file.keys()[0]).value
        numpy.save(os.path.splitext(fileName)[0] + '.npy', data)
