#! /usr/bin/env python3
"""rebin the image with coarser pixel scale
to map the modeled PSF at the instrumental pixel scale
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

__author__ = 'Sebastien Brisard'
__version__ = '1.0.1'
__release__ = __version__


def rebin(a, factor, func=None):
    """Aggregate data from the input array ``a`` into rectangular tiles.

    The output array results from tiling ``a`` and applying `func` to
    each tile. ``factor`` specifies the size of the tiles. More
    precisely, the returned array ``out`` is such that::

        out[i0, i1, ...] = func(a[f0*i0:f0*(i0+1), f1*i1:f1*(i1+1), ...])

    If ``factor`` is an integer-like scalar, then
    ``f0 = f1 = ... = factor`` in the above formula. If ``factor`` is a
    sequence of integer-like scalars, then ``f0 = factor[0]``,
    ``f1 = factor[1]``, ... and the length of ``factor`` must equal the
    number of dimensions of ``a``.

    The reduction function ``func`` must accept an ``axis`` argument.
    Examples of such function are

      - ``numpy.mean`` (default),
      - ``numpy.sum``,
      - ``numpy.product``,
      - ...

    The following example shows how a (4, 6) array is reduced to a
    (2, 2) array

    >>> import numpy
    >>> from rebin import rebin
    >>> a = numpy.arange(24).reshape(4, 6)
    >>> rebin(a, factor=(2, 3), func=numpy.sum)
    array([[ 24,  42],
           [ 96, 114]])

    If the elements of `factor` are not integer multiples of the
    dimensions of `a`, the remainding cells are discarded.

    >>> rebin(a, factor=(2, 2), func=numpy.sum)
    array([[16, 24, 32],
           [72, 80, 88]])

    """
    a = np.asarray(a)
    dim = a.ndim
    if np.isscalar(factor):
        factor = dim*(factor,)
    elif len(factor) != dim:
        raise ValueError('length of factor must be {} (was {})'
                         .format(dim, len(factor)))
    if func is None:
        func = np.mean
    for f in factor:
        if f != int(f):
            raise ValueError('factor must be an int or a tuple of ints '
                             '(got {})'.format(f))

    new_shape = [n//f for n, f in zip(a.shape, factor)]+list(factor)
    new_strides = [s*f for s, f in zip(a.strides, factor)]+list(a.strides)
    aa = as_strided(a, shape=new_shape, strides=new_strides)
    return func(aa, axis=tuple(range(-dim, 0)))
