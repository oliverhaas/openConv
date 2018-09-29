
# openAbel
Python module for fast numerical Abel transforms.


## Table of contents

- [Use case](#use-case)
- [Quick start](#quick-start)
- [Status](#status)
- [Dependencies](#dependencies)
- [Issues](#issues)
- [Algorithms](#algorithms)
- [Copyright and license](#copyright-and-license)


## Use case

The main goal of **openAbel** is to provide fast and efficients Abel transforms of equispaced data
from smooth functions in Python with all actual calculations done in Cython. 
The most useful methods implemented in my module for that purpose use the Fast Multipole Method combined with
arbitrary order end correction of the trapezoidal rule to achieve both fast convergence and linear run time. Other methods are implemented
for comparisons.



## Quick start

In most cases this should be pretty simple:

- Clone the repo: `git clone https://github.com/oliverhaas/openAbel.git`
- Install: `sudo python setup.py install`
- Run example, for example: `python examples/example001_Gaussian.py`

This assumes dependencies are already met and you run a similar system to mine (see [Dependencies](#dependencies)).


## Dependencies

The code was run on several Ubuntu systems without problems. More specific I'm running Ubuntu 16.04 and the following libraries and
Python modules, which were all installed the standard way with either `apt` or `pip`:
- Python 2.7.12
- Numpy 1.14.0
- Scipy 1.1.0
- Cython 0.23.4
- Matplotlib 2.2.0


## Issues

If there are any issues, bugs or feature request just let me know.

## Algorithms

The forward and inverse (or backward) Abel transforms are usually defined as

<a href="https://www.codecogs.com/eqnedit.php?latex=F(y)=2\int_y^\infty\frac{f(r)r}{\sqrt{r^2-y^2}}dr" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?F(y)=2\int_y^\infty\frac{f(r)r}{\sqrt{r^2-y^2}}dr" title="Forward Abel Transform" /></a>
&emsp;&emsp;&emsp;
<a href="https://www.codecogs.com/eqnedit.php?latex=f(r)=-\frac{1}{\pi}\int_r^\infty\frac{F'(y)}{\sqrt{y^2-r^2}}dy" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?f(r)=-\frac{1}{\pi}\int_r^\infty\frac{F'(y)}{\sqrt{y^2-r^2}}dy" title="Backward Abel Transform" /></a>

In this section we will mostly talk about the forward Abel transform and then give some remarks on the inverse Abel transform.

When solved numerically on equispaced data the input data *f(r)* has to be truncated, such that the upper bound of the integral is some
finite value *R*. Usually this isn't a problem since in most cases the functions decay rapidly with *r*. 

It should be noted that often one can use variable transformations or other discretizations to simplify the calculation of the above
integrals. However, often one is interested in exactly the above case due to the 
[relation of the Abel transform with the Fourier and Hankel transforms](https://en.wikipedia.org/wiki/Abel_transform#Relationship_to_the_Fourier_and_Hankel_transforms), 
or by the given data.

There are two obstacles when calculating the transforms numerically: If one wants the output data on the same grid as the input data on *N*
grid points, computational complexity is *O(N^2)*. And the singularity at *r=y* is difficult to handle efficiently.

The first problem can be solved by the [Fast Multipole Method](https://en.wikipedia.org/wiki/Fast_multipole_method) (FMM). The main reference
for the implementation done here was the description of the Chebyshev Interpolation FMM by [Tausch](https://link.springer.com/chapter/10.1007/978-3-642-25670-7_6).
This leads to an *O(N)* algorithm when applied to the (discretized and truncated) Abel transform.

The second problem is often solved by removing the singularity analytically. For example for the Abel transform one can write

<a href="https://www.codecogs.com/eqnedit.php?latex=F(y)=2\int_y^\infty\frac{(f(r)-f(y))r}{\sqrt{r^2-y^2}}dr+f(y)\sqrt{R^2-y^2}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?F(y)=2\int_y^\infty\frac{(f(r)-f(y))r}{\sqrt{r^2-y^2}}dr+f(y)\sqrt{R^2-y^2}" title="Desingularized Forward Abel Transform" /></a>

Now the singularity seems to be removed, but a closer look and one can see that the singularity is still there in the derivative of the
integrand, so the convergence is first order in *N* instead of second order expected when using trapezoidal rule. One can analytically remove the
singularity in higher order with more terms, but for higher order than two the  trapezoidal rule has to be replaced by higher order quadrature rules, which then
usually leads to the [Euler-Maclaurin formula](https://en.wikipedia.org/wiki/Euler%E2%80%93Maclaurin_formula). Since this gets kinda
complicated in higher order (and actually possibly unstable and there are other more elaborate issues) it's simpler 
to go to end corrections which combine handling the singularity and higher order.
These end corrections are described in several publications, but the main reference of **openAbel** was a paper by [Kapur](https://epubs.siam.org/doi/abs/10.1137/S0036142995287847).
If data points outside of the integration interval can be provided these end corrections are arbitrary order stable. Otherwise I wouldn't
recommend going higher than 5th order. As of now we provide the coefficients up to 20th order.
Since the calculation of the end correction end correction coefficients requires some analytical calculations, is quite troublesome and time consuming, 
they have been precalculated in *Mathematica* and stored efficiently, so they only have to be loaded by the **openAbel** code
when needed. The *Mathematica* [notebook](add/calcEndCorr.nb) can be found in this repository as well.

Overall to my knowledge there are no better methods for the described purpose.
For specifically the inverse Abel transform of noisy data there are a lot of algorithms described in literature which perform better in
some aspects, since they either incorporate some assumptions about the data or some kind of smoothing/filtering of the noise. A nice
starting point for people interested in that is the Python module [PyAbel](https://github.com/PyAbel/PyAbel). However, one can use 
**openAbel** for a noisy inverse transform as well, but one should do some manual filtering beforehand. I've had good results with
[maximally flat filters](https://ieeexplore.ieee.org/document/7944698/) (see [example003](examples/example003_inverse.py) 
and the *Mathematica* [notebook](add/calcMaxFlat.nb)).

In the examples some more details are discussed and mentioned; in general the examples are a good way to learn how to understand and
use the code.


## Copyright and license

The code **openAbel** is published under the GNU GPL version 3. For more information see [COPYING.md](COPYING.md). 
Since I'm mostly interested in getting reasonable credit for my work, you can write me if you need some other arrangement.





