
openConv README
=========

.. image:: https://travis-ci.org/oliverhaas/openConv.svg?branch=master
    :target: https://travis-ci.org/oliverhaas/openConv
    :alt: Build Status

.. image:: https://readthedocs.org/projects/openconv/badge/?version=latest
    :target: https://openconv.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

**Note:** It's best to view this readme in the 
**openConv** `documentation <https://openconv.readthedocs.io/en/latest/index.html>`_.



Introduction
--------------


The main goal of **openConv** is to provide fast and efficient numerical convolutions of symmetric
and smooth kernels and data of equispaced data in Python with all actual calculations done in Cython. 
The most useful methods implemented in my module for that purpose use the Fast Multipole Method combined with
arbitrary order end correction of the trapezoidal rule to achieve both fast convergence and linear run time. Other methods are implemented
for comparisons.



Quick Start
--------------


In most cases this should be pretty simple:

- Clone the repository: :code:`git clone https://github.com/oliverhaas/openConv.git`
- Install: :code:`sudo python setup.py install`
- Run example: :code:`python example000_XXX.py`

This assumes dependencies are already met and you run a more or less similar system to mine (see `Dependencies`_).



Dependencies
--------------

The code was run on several Ubuntu systems without problems. More specific I'm running Ubuntu 16.04 and the following libraries and
Python modules, which were all installed the standard way with either :code:`sudo apt install libName` or 
:code:`sudo pip install moduleName`. 

- Python 2.7.12

- Numpy 1.14.0

- Scipy 1.1.0

- Cython 0.23.4

- Matplotlib 2.2.0

- gcc 5.4.0

- FFTW3 3.3.4


As usual newer versions of the libraries should work as well, and many older versions will too. I'm sure it's possible to
get **openConv** to run on vastly different systems, like e.g. Windows systems, but obviously I haven't extensively tested
different setups.



Issues
--------------


In contrast to other codes I made available, **openConv** has as of now only very specific use-cases I actually needed, thus implemented and debugged. I strongly recommend every user to thourougly check if the methods work as intended for their specific problem. For most people **openConv** will thus not be a useable code as is, but more a starting point or inspriration for their own code.
If there are any issues, bugs or feature request just let me know. Gaps in the implementation might be filled by me if requested.



Transform Methods
--------------


It is fairly common to use directly the discrete convolution to approximate the convolution integral, often with smaller
improvements like using trapezoidal rule instead of rectangle rule. This yields usually neither good order of convergence
(second order with trapezoidal rule), nor fast calculation (quadratic *O(N^2)* computational complexity). **openConv** intends to provide methods to calculate these convolutions efficiently, fast, and with high accuracy. Beside the common "fast convolution" algorithm based on the Fast Fourier Transform we provide methods based on the Fast Multipole Method and high order end correction, which outclass common methods in many cases in most aspects (convergence order, error, computational complexity, etc.).

In the documentation and the examples more details are discussed and mentioned; in general both are a good way to learn how to understand and use the code.


Copyright and License
--------------

Copyright 2016-2018 Oliver Sebastian Haas.

The code **openConv** is published under the GNU GPL version 3. This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

For more information see the GNU General Public License copy provided in this repository `LICENSE <https://github.com/oliverhaas/openAbel/tree/master/LICENSE>`_.












