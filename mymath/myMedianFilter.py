import numpy as np


def myMedianFilter(im, size):
    """My median filter function. np.nanmedian is used to correctly deal
    with nan, comparing to medfilt in scipy, which ignores nan.

    :param im: input image, 2D np.array
    :param size: size of the median filter, should be a 2 element array
    :returns: filtered image
    :rtype: 2D np.array
    """

    nImage = size[0] * size[1]
    imCube = np.empty((im.shape[0], im.shape[1], nImage), dtype='float')
    im0 = im.data.copy()
    im0[im.mask > 0] = np.nan
    for i in range(size[0]):
        for j in range(size[1]):
            shift_i = i - size[0] // 2
            shift_j = j - size[1] // 2
            imCube[:, :, i * size[1] + j] = np.roll(
                im0, [shift_i, shift_j], axis=(0, 1))
    return np.nanmedian(imCube, axis=2)
