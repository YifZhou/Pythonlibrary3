#! /usr/bin/env python
"""calcualte the flat field for grism observations
"""


from astropy.io import fits
import numpy as np


def xtowl(x):
    """
    convert the pixel index to wavelength
    """
    return (x * 46.7 + 8953.)


def calFlat(x0, outSize=256):
    """
    x0 the x corrdinate of the source, in detector coordinate
    """
    # read in the flat cube
    flatCube = np.zeros((1014, 1014, 4))
    for i in range(4):
        flatCube[:, :, i] = fits.getdata('WFC3.IR.G141.flat.2.fits', i)
    lambda_min = 10600
    lambda_max = 17000
    lambdaList = xtowl(np.arange(1014) - x0)
    flat = flatCube[:, :, 0].copy()
    l = (lambdaList - lambda_min) / (lambda_max - lambda_min)
    for i in range(1014):
        flat[:, i] = flat[:, i] + flatCube[:, i, 1] * l +\
                     flatCube[:, i, 2]*l**2 + flatCube[:, i, 3]*l**3
    margin = (1014 - outSize) // 2
    return flat[margin:-margin, margin:-margin]

# x0List = [19.915, 23.95, 20.06, 20.16, 20.06, 21.91, 21.94, 23.26, 23.94,
#           21.73, 21.73, 21.73, 21.16, 24.11,
#           24.16]  # x coord of the direct image


# flatCube = np.zeros([256, 256, 4])
