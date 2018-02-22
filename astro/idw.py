# cython: profile=True, boundscheck=False, nonecheck=False, wraparound=False
# cython: cdivision=True
from idwinterp import clean_idwinterp


def idw(im, crmask, mask):
    """wrap the c code

    :param im: input image
    :param crmask: cosmic ray mask
    :param mask: bad pixel mask
    :returns: corrected image
    :rtype: np.array

    """
    imcorr = im.copy().astype(float)
    nx = im.shape[1]
    ny = im.shape[0]
    print(type(imcorr))
    clean_idwinterp(imcorr, crmask.astype(bool), mask.astype(bool),
                    nx, ny, 0)
    return imcorr
