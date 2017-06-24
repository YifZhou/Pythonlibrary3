#! /usr/bin/env python

"""modified 2D gaussian kernel
"""

import math

import numpy as np
from astropy.convolution import Kernel2D
from astropy.modeling.functional_models import Gaussian2D


def _round_up_to_odd_integer(value):
    i = int(math.ceil(value))  # TODO: int() call is only needed for six.PY2
    if i % 2 == 0:
        return i + 1
    else:
        return i


class Gaussian2DKernel(Kernel2D):
    """
    2D Gaussian filter kernel.

    The Gaussian filter is a filter with great smoothing properties. It is
    isotropic and does not produce artifacts.

    Parameters
    ----------
    stddev : number
        Standard deviation of the Gaussian kernel.
    x_size : odd int, optional
        Size in x direction of the kernel array. Default = 8 * stddev.
    y_size : odd int, optional
        Size in y direction of the kernel array. Default = 8 * stddev.
    mode : str, optional
        One of the following discretization modes:
            * 'center' (default)
                Discretize model by taking the value
                at the center of the bin.
            * 'linear_interp'
                Discretize model by performing a bilinear interpolation
                between the values at the corners of the bin.
            * 'oversample'
                Discretize model by taking the average
                on an oversampled grid.
            * 'integrate'
                Discretize model by integrating the
                model over the bin.
    factor : number, optional
        Factor of oversampling. Default factor = 10.


    See Also
    --------
    Box2DKernel, Tophat2DKernel, MexicanHat2DKernel, Ring2DKernel,
    TrapezoidDisk2DKernel, AiryDisk2DKernel, Moffat2DKernel

    Examples
    --------
    Kernel response:

     .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.convolution import Gaussian2DKernel
        gaussian_2D_kernel = Gaussian2DKernel(10)
        plt.imshow(gaussian_2D_kernel, interpolation='none', origin='lower')
        plt.xlabel('x [pixels]')
        plt.ylabel('y [pixels]')
        plt.colorbar()
        plt.show()

    """
    _separable = True
    _is_bool = False

    def __init__(self, x_stddev, y_stddev, rho=0, **kwargs):
        cov = np.array([[x_stddev**2, rho * x_stddev * y_stddev],
                        [rho * x_stddev * y_stddev, y_stddev**2]])
        self._model = Gaussian2D(1. / (2 * np.pi * x_stddev * y_stddev * np.sqrt(1-rho**2)), 0,
                                        0, cov=cov)
        self._default_size = _round_up_to_odd_integer(8 * max(x_stddev, y_stddev))
        super(Gaussian2DKernel, self).__init__(**kwargs)
        self._truncation = np.abs(1. - self._array.sum())
