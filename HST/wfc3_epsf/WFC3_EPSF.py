# /usr/bin/env python3

"""implement intrpolation algorithms for interpolating method for
WFC3 Empirical PSFs
"""

import numpy as np
from astropy.io import fits


class WFC3_EPSF:
    """ WFC3 Empirical PSFs
    """
    def __init__(self, filterName, NX=3, NY=3,
                 XList=(0, 507, 1014), YList=(0, 507, 1014)):
        pklFN = ''
        self.PSFCube = fits.getdata(fitsFN)
        self.nPSFs = self.PSFCube.shape[0]
        self.PSFshape = self.PSFCube.shape[1:]
        self.NX = NX
        self.NY = NY
        xx, yy = np.meshgrid(XList, YList, indexing='ij')
        self.XSep = XList[1] - XList[0]
        self.YSep = YList[1] - YList[0]
        self.XList = xx.flatten()
        self.YList = yy.flatten()
        self.superSamp = 4  # super sample rate
        self.x = 507
        self.y = 507

    def interp(self, x, y, subframe=256):
        self.x = x + (1014 - subframe) / 2
        self.y = y + (1014 - subframe) / 2
        Xfrac = 1 - np.abs(self.XList - self.x) / self.XSep
        Xfrac[Xfrac < 0] = 0
        Yfrac = 1 - np.abs(self.YList - self.y) / self.YSep
        Yfrac[Yfrac < 0] = 0
        PSF = np.zeros(self.PSFshape)
        for i in range(self.nPSFs):
            PSF += self.PSFCube[i, :, :] * Xfrac[i] * Yfrac[i]
        return PSF


if __name__ == '__main__':
    fn = './PSFSTD_WFC3IR_F105W.fits'
    epsf = WFC3_EPSF(fn)
    p = epsf.interp(128, 128)
    from plot import imshow
    imshow(p)
