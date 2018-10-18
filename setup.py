from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ecadef = ["-O3", "-Wunused-but-set-variable"]
#ecadef = ["-O0", "-g", "-Wunused-but-set-variable"]
iddef = ["/usr/local/include/", "./", numpy.get_include()]
lddef = ["/usr/local/lib/"]
compdir = {'boundscheck': False, 'nonecheck': False, 'wraparound': False, 'cdivision': True, 'profile': False, 'infer_types': False}

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


setup(name = 'openConv',
      version='0.1',
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
                                  
              
openConv  Copyright (C) 2017-2018  Oliver Sebastian Haas
                                             
"""
print logoArt
     
     
