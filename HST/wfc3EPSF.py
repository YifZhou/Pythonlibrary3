# /usr/bin/env python3

"""implement intrpolation algorithms for interpolating method for
WFC3 Empirical PSFs
"""

import numpy as np
import pickle
import os
from os import path
from scipy.ndimage.interpolation import shift, zoom
from image import rebin
fDIR = os.path.dirname(os.path.abspath(__file__))


class WFC3_EPSF:
    """ WFC3 Empirical PSFs
    """
    def __init__(self, filterName, x0, y0, subframe=256, NX=3, NY=3,
                 XList=(0, 507, 1014), YList=(0, 507, 1014)):
        pklFN = path.expanduser(path.join(fDIR, 'wfc3_epsf',
                                          'PSFSTD_WFC3IR_{0}.pkl'.format(filterName)))
        with open(pklFN, 'rb') as pkl:
            self.PSFCube = pickle.load(pkl)
        self.nPSFs = self.PSFCube.shape[0]
        self.PSFshape = self.PSFCube.shape[1:]
        self.subframe = subframe
        self.x = x0 + (1014 - subframe) / 2
        self.y = y0 + (1014 - subframe) / 2  # convert to detector frame
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
        self.interp()

    def interp(self):
        Xfrac = 1 - np.abs(self.XList - self.x) / self.XSep
        Xfrac[Xfrac < 0] = 0
        Yfrac = 1 - np.abs(self.YList - self.y) / self.YSep
        Yfrac[Yfrac < 0] = 0
        self.PSF0 = np.zeros(self.PSFshape)
        for i in range(self.nPSFs):
            self.PSF0 += self.PSFCube[i, :, :] * Xfrac[i] * Yfrac[i]
        return self.PSF0

    def getPSF(self, dx, dy, z=1):
        """
        add a zoom parameter to adjust for focus changing
        """
        PSFi = zoom(self.PSF0, z)
        shiftedPSF = shift(PSFi, (dy*self.superSamp - 0.5, dx*self.superSamp - 0.5))
        PSF0 = rebin(shiftedPSF[:100, :100], 4, func=np.mean)
        return PSF0 / PSF0.sum()   # return the normalized PSF


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    epsf = WFC3_EPSF('F105W', 128, 128)
    p = epsf.getPSF(0, 0)
    from plot import imshow
    imshow(p, vmax=0.01)
    plt.show()
