Transform Methods
=================


In **openAbel** there are a couple of different algorithms for the calculation of the Abel transforms implemented, although most of them are just for comparisons and is it recommended to only use the default method. 

Due to the equispaced discretization all methods truncate the Abel transform integral, e.g. for the forward Abel transform 

.. math::
        F(y)=2\int_y^\infty\frac{f(r)r}{\sqrt{r^2-y^2}}dr\approx2\int_y^R\frac{f(r)r}{\sqrt{r^2-y^2}}dr\; .
        
This is sometimes called finite Abel transform. Since :math:`f(r)` often has compact support or decays very quickly (and R can
be chosen very large with a fast transform method) this usually introduces an arbitrary small error.

It should be noted that often one can use variable transformations or other discretizations to simplify the calculation of the above
integrals. However, often one is interested in exactly the in **openAbel** implemented case on equispaced discretization. This is often due to the 
`relation of the Abel transform with the Fourier and Hankel transforms <https://en.wikipedia.org/wiki/Abel_transform#Relationship_to_the_Fourier_and_Hankel_transforms>`_ and the desire to use the same discretization as the FFT or a discrete convolution, or just by the given data (e.g. from experiments).

The main two obstacles when calculating the transforms numerically are the singularity at :math:`r=y` and the dependence of the 
result on :math:`y`, meaning computational complexity is quadratic :math:`O(N^2)` if one naively integrates. The main difference
between the implemented transforms is how those two issues are treated.

When creating the Abel transform object the :code:`method` keyword argument can be provided to chose different transform methods:

.. code-block:: python

    import openAbel
    abelObj = openAbel.Abel(nData, forwardBackward, shift, stepSize, method = 3, order = 2)

The methods with end corrections can do the transformation in different orders of accuracy by setting
the :code:`order` keyword argument; all other methods ignore :code:`order`. Note when we
talk about :math:`n` order accuracy we usually mean :math:`(n+1/2)` order accuracy due to the square
root in the Abel transform kernel. For higher order methods the transformed function has to be sufficiently
smooth to achieve the full order of convergence, and in very extreme cases the transform become unstable if
high order is used on non-smooth functions.
The length of the data vector :code:`nData` we denote as :math:`N` in the math formulas.

Overall cases where a user should use anything other than :code:`method = 3` (default) and :code:`order = 2` (default) to :code:`order = 5`
will be very rare. For a detailed comparison of the methods it is recommended to look at 
`example004_fullComparison <https://openabel.readthedocs.io/en/latest/examples/example004.html>`_.


Desingularized Trapezoidal Rule
--------------

.. code-block:: python

    # order keyword argument is ignored (only first order implemented)
    abelObj = openAbel.Abel(nData, forwardBackward, shift, stepSize, method = 0)    

The desingularized trapezoidal rule is probably the simplest practicable algorithm. 
It subtracts the singularity and integrates it analytically, and numerically integrates the 
remaining desingularized term with the trapezoidal rule. In the implementation this is done to first order, i.e. for
the forward Abel transform this leads to

.. math::
        F(y)=2\int_{y}^{R}\frac{(f(r)-f(y))r}{\sqrt{r^2-y^2}}dr+f(y)\sqrt{R^2-y^2}\;.
        
Now the singularity seems to be removed, but a closer look and one can see that the singularity
is still there in the derivative of the integrand, so the convergence is first order in :math:`N`
instead of second order expected when using trapezoidal rule. One can analytically remove the
singularity in higher order with more terms, but this gets kinda complicated 
(and possibly unstable, plus there are other practical issues). The trapezoidal rule portion of the method 
leads to quadratic :math:`O(N^2)` computational complexity of the method.


Hansen-Law Method
--------------

.. code-block:: python

    # order keyword argument is ignored (only somewhat first order implemented)
    abelObj = openAbel.Abel(nData, forwardBackward, shift, stepSize, method = 1)    
    
The Hansen-Law method by `Hansen and Law <https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-2-4-510>`_ 
is a space state model approximation of the Abel transform kernel.
With that method recursively transforms a piecewise linear approximation of the input functions 
to integrate analytically piece by piece. In principle this results in an 2nd order accurate
transform, but the approximation of the Abel transform kernel as a sum of exponentials is quite difficult.
In other words the approximation

.. math::
        \frac{1}{\sqrt{1-\exp{(-2t)}}}\approx\sum_{k=1}^K\exp{(-\lambda_kt)} 
    
is in practice not possible to achieve with high accuracy and reasonable :math:`K`. This is the main 
limitation of the method, and the original space state model approxmation has a typical relative
error of :math:`10^{-3}` at best -- then it just stops converging with increasing :math:`N`. If one 
ignores several details that makes the method apparently linear :math:`O(N)` computational complexity,
so it is implemented here for comparisons.


Trapezoidal Rule with End Corrections
--------------

.. code-block:: python

    # 0 < order < 20
    abelObj = openAbel.Abel(nData, forwardBackward, shift, stepSize, method = 2, order = 2)

The trapezoidal rule with end correction improves on the desingularized trapezoidal rule.
It doesn't require analytical integration because it uses precalculated end correction coefficients
of arbitrary order. As described in `Kapur <https://epubs.siam.org/doi/abs/10.1137/S0036142995287847>`_
one can contruct :math:`\alpha_i` and :math:`\beta_i` such that the approxmation

.. math::
        \int_{a}^{b}f(x)dx \approx h\cdot\sum_{i=1}^{N-2}f(x_i) + 
                                   h\cdot\sum_{i=0}^{M-1}\alpha_if(x_{i-p}) + 
                                   h\cdot\sum_{i=0}^{M-1}\beta_if(x_{N-1-q})

is accurate to order :math:`M`. Note that :math:`p` and :math:`q` should be chosen such that the correction is
centered around the end points: Similar to central finite differences this leads to an arbitrary order stable scheme,
and thus incredibly fast convergence and small errors.
Otherwise it's not recommended to go higher than :math:`M=5`, again similar to forward and backward finite
differences. The trapezoidal rule portion of the method leads to quadratic :math:`O(N^2)` computational
complexity of the method.

Since the calculation of the end correction coefficients requires some analytical calculations, is quite troublesome and time consuming, 
they have been precalculated in *Mathematica* and stored in binary *\*.npy* , so they are only loaded by the **openAbel** code
when needed and don't have to be calculated. 
The `*Mathematica* notebook <https://github.com/oliverhaas/openAbel/tree/master/add/calcEndCorr.nb>`_ which was 
used to calculate these end correction coefficients can be found in this repository as well.



Fast Multipole Method with End Corrections
--------------

.. code-block:: python

    # 0 < order < 20
    abelObj = openAbel.Abel(nData, forwardBackward, shift, stepSize, method = 3, order = 2)

The default and recommended method is the Fast Multipole Method (FMM) with end corrections. This method provides a fast
linear :math:`O(N)` computational complexity transform of arbitrary order.
The specific FMM used is based on Chebyshev interpolation and nicely described
and applied by `Tausch <https://link.springer.com/chapter/10.1007/978-3-642-25670-7_6>`_ on a similar problem.
In principle the FMM uses a hierarchic decomposition to combine a linear amount of direct short-range contributions
and smooth approximations of long-range contributions with efficient reuse of intermediate results to get in total 
a linear :math:`O(N)` computational complexity algorithm. This method thus provides extremely fast convergence and
fast computational, and is optimal for the intended purpose.



Remarks on Transforms of Noisy Data
--------------

For specifically the inverse Abel transform of noisy data there are a lot of algorithms described in literature which perform better in
some aspects, since they either incorporate some assumptions about the data or some kind of smoothing/filtering of the noise. A nice
starting point for people interested in those methods is the Python module `PyAbel <https://github.com/PyAbel/PyAbel>`_. 

However, there is no reason not to combine the methods provided in **openAbel** with some kind of filering for nicer results.
I've had good results with`maximally flat filters <https://ieeexplore.ieee.org/document/7944698/>_, as seen
in `example003_noisyBackward <https://openabel.readthedocs.io/en/latest/examples/example003.html>`_, and with additional material
in the `Mathematica notebook <https://github.com/oliverhaas/openAbel/tree/master/add/calcEndCorr.nb>`_.


