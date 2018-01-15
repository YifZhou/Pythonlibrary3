import numpy as np
from mymath import windowed2D_std, myMedianFilter


def cosmics(im, err, dq, median, window=7, sigma=5, nIter=6):
    """Use iterative 2D median filter to recognize cosmic rays in the

    :param im: input image
    :param err: error image
    :param dq: data quality mask
    :param median: median combined frame. Used for come up with
    correct median filter
    :param window: (default: 7) window size
    :param sigma: (default: 5) threshold of n sigma
    :param nIter: (default: 6) number of iterations

    """
    # initialize the cosmic ray masks
    crMask = np.zeros_like(dq)
    # err1: extra error considering the median filters, important when
    # image has large gradient
    # err0: combination of intrinsic errors
    for i in range(nIter):
        imMasked = np.ma.MaskedArray(im, dq + crMask)
        err1_1 = np.abs(windowed2D_std(median, widths=[1, window],
                                       mask=dq)) / np.sqrt(window)
        err1_2 = np.abs(windowed2D_std(median, widths=[window, 1],
                                       mask=dq)) / np.sqrt(window)
        diff1 = (imMasked - myMedianFilter(imMasked, [1, window])) /\
                np.sqrt(err**2 + err1_1**2)
        diff2 = (imMasked - myMedianFilter(imMasked, [window, 1])) /\
                np.sqrt(err**2 + err1_2**2)
        diff = np.minimum(np.abs(diff1), np.abs(diff2))
        yCR, xCR = np.where(diff > sigma)
        crMask[yCR, xCR] = 1
        crMask = crMask - dq
        crMask[crMask < 0] = 0
        neighbors = np.roll(crMask, (1, 0), axis=(0, 1)) +\
                                                  np.roll(crMask, (-1, 0), axis=(0, 1)) +\
                                                  np.roll(crMask, (0, 1), axis=(0, 1)) +\
                                                  np.roll(crMask, (0, -1), axis=(0, 1))
        crMask[neighbors > 1] = 1
    yCR, xCR = np.where(crMask == 1)
    return crMask, xCR, yCR
