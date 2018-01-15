#! /usr/bin/env python

from os import path
import numpy as np
from astropy.io import fits
"""create a mask for sky subtraction
"""


def scanMask(mask0, masks):
    """create mask to maskout the region with scanning spectra

    :param mask0: input mask, normally are all zeros array
    :param masks: scanning occupied region, [x0, x1, y0, y1]

    """
    for mask in masks:
        mask0[mask[1]:mask[3], mask[0]:mask[2]] = 1
    return mask0


if __name__ == '__main__':
    pass
    # masks = [
    #     [[60, 150, 188, 235], [100, 39, 224, 120], [76, 50, 199, 136]],
    #     [[66, 144, 190, 239], [68, 26, 193, 128], [50, 49, 174, 136]],
    #     [[60, 152, 185, 234], [106, 38, 229, 124], [82, 52, 204, 135]],
    #     [[60, 150, 185, 224], [117, 42, 236, 129], [91, 55, 220, 140]],
    #     [[60, 150, 185, 235], [117, 44, 243, 129], [90, 56, 218, 139]],
    #     [[65, 144, 186, 242]], [[66, 143, 183, 240]], [[66, 154, 189, 234]],
    #     [[65, 142, 190, 242], [56, 28, 179, 124]], [[64, 144, 194, 239]],
    #     [[65, 142, 193, 239]], [[66, 144, 199, 240]], [[64, 144, 189, 237]],
    #     [[65, 144, 189, 242], [39, 34, 161, 129]],
    #     [[69, 142, 190, 240], [66, 30, 187, 126]]
    # ]  # mask for 15 visits
    # for visit in range(15):
    #     saveFN = 'skyMask_visit_{0:02d}.fits'.format(visit+1)
    #     mask0 = np.ones((256, 256))
    #     mask=makeMask(mask0, masks[visit])
