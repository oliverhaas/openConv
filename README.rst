
openConv README
=========


.. image:: https://travis-ci.org/oliverhaas/openConv.svg?branch=master
    :target: https://travis-ci.org/oliverhaas/openConv

**Note:** It's best to view this readme in the 
`openConv documentation <https://openconv.readthedocs.io/en/latest/index.html>`_



Introduction
--------------


The main goal of **openConv** is to provide fast and efficients numerical convolutions of symmetric
and smooth kernels and data of equispaced data from smooth functions in Python with all actual calculations done in Cython. 
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


As usual newer versions of the libraries should work as well, and many older versions will too. I'm sure it's possible to
get **openConv** to run on vastly different systems, like e.g. Windows systems, but obviously I haven't extensively tested
different setups.



Issues
--------------

If there are any issues, bugs or feature request just let me know. As of now there are some gaps in the implementation, e.g.
not all convolution symmetry types are available in all methods, but can be added if requested.



Transform Methods
--------------


The convolution integral in one dimension is defined as

<a href="https://www.codecogs.com/eqnedit.php?latex(f*g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?(f*g)(t)=\int_{-\infty}^{\infty}f(\tau)g(t-\tau)d\tau" title="Convolution Integral" /></a>

and the discrete equivalent (which implies uniformly discretized data and kernel)

<a href="https://www.codecogs.com/eqnedit.php?latex(f*g)[n]=\sum_{m=-\infty}^{\infty}f[m]g[n-m]" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?(f*g)[n]=\sum_{m=-\infty}^{\infty}f[m]g[n-m]" title="Convolution Discrete" /></a>

Since both f and g often have some kind of symmetry around 0, **openConv** deals only with that case for now.

It is fairly common to use directly the discrete convolution to approximate the convolution integral, often with smaller
improvements like using trapezoidal rule instead of rectangle rule as above. This yields usually neither good order of convergence
(second order with trapezoidal rule), nor fast calculation (quadratic *O(N^2)* computational complexity).

One option instead of direct calculation is the use of an Fast Fourier Transform (FFT) based convolution, or often called
"fast convolution" algorithm. This approach gives linearithmic *O(Nlog(N))* computational complexity, but possibly large
relative errors of the result, since the error scales with the maximum values of all the kernel and data values. In case of
functions with high dynamic range, e.g. Gaussians, the tails of the results are poorly resolved.

Even better computational complexity (linear *O(N)*) can be achieved by the [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method) (FMM). 
There are so called black box FMM described in literature (e.g. by [Tausch](https://link.springer.com/chapter/10.1007/978-3-642-25670-7_6)),
which in principle work well for smooth kernels with not too high dynamic range. In **openConv** we extend this scheme to functions
with somewhat exponential decay and thus can deal with a large class of functions with high dynamic range.

To increase the order of convergence **openConv** uses end corrections for the trapezoidal rule as described in the 
reference by [Kapur](https://epubs.siam.org/doi/abs/10.1137/S0036142995287847).
If data points outside of the integration interval can be provided these end corrections are arbitrary order stable. Otherwise I wouldn't
recommend going higher than 5th order. As of now we provide the coefficients up to 20th order. The *Mathematica* 
[notebook](add/calcCoeffsSmooth.nb) which calculated these coefficients can be found in this repository as well.

In the examples some more details are discussed and mentioned; in general the examples are a good way to learn how to understand and
use the code.


Copyright and License
--------------

Copyright 2016-2018 Oliver Sebastian Haas.

The code **openConv** is published under the GNU GPL version 3. This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

For more information see the GNU General Public License copy provided in this repository `LICENSE <https://github.com/oliverhaas/openAbel/tree/master/LICENSE>`_.











