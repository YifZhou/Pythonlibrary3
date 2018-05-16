#! /usr/bin/env python
import pickle
from os import path
import numpy as np

""" get field dependent G141 master sky original sky cube from WFC3
ISR 2015-17

in WFC3 ISR 2015-17, the sky cube is flat flieded, the function here
returns the sky cube with flat field multiplied for sky subtraction
"""

scriptDIR = path.dirname(path.realpath(__file__))


def G141SkyCube(subframe=256):
    """calculate the field dependent sky cube

    :param subframe: size of the subframe for output
    """
    with open(path.join(scriptDIR, 'G141_skycube.pkl'), 'rb') as pkl:
        sky = pickle.load(pkl)
    outsky = np.zeros((subframe, subframe, 3))
    margin = (1014 - subframe) // 2
    for i in range(3):
        if margin != 0:
            outsky[:, :, i] = sky[margin:-margin, margin:-margin, i]
        else:
            outsky[:, :, i] = sky[:, :, i]
    return outsky
