#! /usr/bin/env python
"""use webbpsf package to calcualte JWST PSFs
"""

import webbpsf


def calc_psf(filterName, oversample=4, offset_r=0, offset_theta=0, instrument='nircam'):
    """a convenieint function for using webbpsf to create JWST PSFs

    :param filterName: Name fo the filter
    :param offset_r: angular separation between the host and the companion
    :param offset_theta: PA between the host and the companion
    :param instrument:n name of the used instrument
    :returns: generated JWST psf
    :rtype: HDUList

    """
    # TODO complete instrument selection list
    if instrument.lower() == 'nircam':
        instr = webbpsf.NIRCam()
    instr.filter = filterName
    instr.options['source_offset_r'] = offset_r
    instr.options['source_offset_theta'] = offset_theta
    PSF = instr.calc_psf(oversample=oversample)
    # retern the oversampled data
    return PSF[0].data
