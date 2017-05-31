#! /usr/bin/env python3

"""convert standard PSF fits to pickle for python file to pick it up
"""

from astropy.io import fits
import pickle
import glob

fitsFNs = glob.glob('*.fits')

for fn in fitsFNs:
    dcube = fits.getdata(fn, 0)
    with open(fn.replace('.fits', '.pkl'), 'wb') as pkl:
        pickle.dump(dcube, pkl)
