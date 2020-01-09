from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import cython
import numpy
import scipy
import matplotlib
import sys
import os

print('\nDependencies versions:')
print('python: ' + str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2]))
print('cython: ' + cython.__version__)
print('numpy: ' + numpy.__version__)
print('scipy: ' + scipy.__version__)
print('matplotlib: ' + matplotlib.__version__)
print()

ecadef = ["-O3", "-Wunused-but-set-variable"]
iddef = ["/usr/local/include/", "./", numpy.get_include()]
lddef = ["/usr/local/lib/"]
compdir = {'boundscheck': False, 'nonecheck': False, 'wraparound': False, 'cdivision': True, 
           'profile': False, 'infer_types': False, 'language_level' : '3'}

extensions = cythonize([
                        Extension('openConv.base',
                            sources=['openConv/base.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.trap',
                            sources=['openConv/trap.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.fft',
                            sources=['openConv/fft.pyx'],
                            extra_compile_args = ecadef + ['-lfftw3', '-lm'],
                            extra_link_args = ['-lfftw3', '-lm'],
                            include_dirs = iddef
                            ),
                        Extension('openConv.fmm',
                            sources=['openConv/fmm.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.wrap',
                            sources=['openConv/wrap.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.constants',
                            sources=['openConv/constants.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.mathFun',
                            sources=['openConv/mathFun.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.interpolate',
                            sources=['openConv/interpolate.pyx'],
                            extra_compile_args = ecadef,
                            include_dirs = iddef
                            ),
                        Extension('openConv.cheb',
                            sources=['openConv/cheb.pyx'],
                            extra_compile_args = ecadef + ['-lfftw3', '-lm'],
                            extra_link_args = ['-lfftw3', '-lm'],
                            include_dirs = iddef
                            )
                        ], 
                        compiler_directives = compdir
                        )

vers = '0.2'
setup(name = 'openConv',
      version=vers,
      packages = ['openConv'],
      package_data={'openConv': ['*.pxd','coeffsData/*']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = extensions
     )

logoArt = """

                             ___             
          ___ _ __  ___ _ _ / __|___ _ ___ __
         / _ \ '_ \/ -_) ' \ (__/ _ \ ' \ V /
         \___/ .__/\___|_||_\___\___/_||_\_/ 
             |_|                             
                                  
              
openConv """+vers+"""  Copyright (C) 2017-2020  Oliver Sebastian Haas
                                             
"""
print(logoArt)
     
     
