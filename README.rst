
openAbel README
=========

.. image:: https://travis-ci.org/oliverhaas/openAbel.svg?branch=master
    :target: https://travis-ci.org/oliverhaas/openAbel

**Note:** It's best to view this readme in the 
`openAbel documentation <https://openabel.readthedocs.io/en/latest/index.html>`_



Introduction
--------------


The main goal of **openAbel** is to provide fast and efficients Abel transforms of equispaced data
from smooth functions in Python with all actual calculations done in Cython. 
The most useful methods implemented in this module for that purpose use the Fast Multipole Method combined with
arbitrary order end correction of the trapezoidal rule to achieve small errors and fast convergence,
as well as linear computational complexity. A couple of other methods are implemented for comparisons.
Abel transform function can be used from Python and with numpy arrays or from Cython using pointers.



Quick Start
--------------

In most cases this should be pretty simple:

- Clone the repository: :code:`git clone https://github.com/oliverhaas/openAbel.git`
- Install: :code:`sudo python setup.py install`
- Run example: :code:`python example000_simpleForward.py`

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
get **openAbel** to run on vastly different systems, like e.g. Windows systems, but obviously I haven't extensively tested
different setups.



Issues
--------------

If there are any issues, bugs or feature request just let me know. As of now there are some gaps in the implementation, e.g.
not all transform types are available in all methods, but since the default method vastly outperforms every other method 
anyway it's not really a pressing issue.



Transform Methods
--------------

For the default and most important method of **openAbel** we adapted the Chebyshev interpolation Fast Multipole Method (FMM) 
as described by `Tausch <https://link.springer.com/chapter/10.1007/978-3-642-25670-7_6>`_ and calculated end corrections 
specifically for the Abel transform similar to `Kapur <https://epubs.siam.org/doi/abs/10.1137/S0036142995287847>`_. 
If data points outside of the integration interval can be provided these end corrections are arbitrary order stable, otherwise
it's recommended to use at most 5th order. The FMM leads to an linear *O(N)* computational complexity algorithm.

In both error computational complexity there is no better method for the intended purpose to my knowledge.

For more information see the openAbel documentation on the 
`transform methods <https://openabel.readthedocs.io/en/latest/transformMethods.html>`_ and 
`examples <https://openabel.readthedocs.io/en/latest/examples.html>`_.


Copyright and License
--------------

Copyright 2016-2018 Oliver Sebastian Haas.

The code **openAbel** is published under the GNU GPL version 3. This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation. 

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

For more information see the GNU General Public License copy provided in this repository `LICENSE <https://github.com/oliverhaas/openAbel/tree/master/LICENSE>`_.












