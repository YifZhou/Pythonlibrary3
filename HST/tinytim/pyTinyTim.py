#! /usr/bin/env python
from __future__ import print_function
import subprocess as sp
import os

#  import sys
"""
Use Tinytim to generate a PSFs
"""

__all__ = ['pyTinyTim']


def pyTinyTim(xc, yc, fileName, jitx=0, jity=0, secDis=0,
              outputRoot=None, outputDIR='.', silent=False):
    """use a pre-defined configuration file generated by tiny1 to
              generate the PSF file. Python script will run tiny2 and
              tiny3 to produce tinytim fits file

    :param xc: x centroid of the psf in detector coordinate
    :param yc: y centroid of the psf in detector coordinate
    :param fileName: input tiny1 file name
    :param jitx: jittering in x
    :param jity: jittering in y direction, these two factor would slightly change the FWHM of the PSF in two direction
    :param secDis: the displacement of the secondary in micron. The
    general shrinkage of HST and breathing both would change this
    parameter slightly
    :param outputRoot: root file name of the output tinytim fits file,
    if not specified, root name will by determined by the jitter,
    displacement and x/y location
    :param outputDIR: output location of the tinytim PSF
    :param silent:
    :returns: file name of the tinytim fits file
    :rtype: string

    """

    if outputRoot is None:
        outputRoot = 'Jitx_{0:0>2d}_Jity_{1:0>2d}_Dis_{2:0>2.2f}_x_{3:d}_y_{4:d}_'.format(
            jitx, jity, secDis, xc, yc)

    # Modify the configuration file
    inFile = open(fileName, 'r').readlines()
    inFile[1] = outputRoot + '\n'
    inFile[9] = '{0:.5f} # Major axis jitter in mas\n'.format(jitx)
    inFile[10] = '{0:.5f} # Minor axis jitter in mas\n'.format(jity)
    inFile[13] = '{0:d} {1:d}  # Position 1\n'.format(int(xc), int(yc))
    inFile[-19] = '{0:.5f} #z4 = Focus\n'.format(secDis * 0.011)
    with open('temp.in', 'w') as out:
        out.writelines(inFile)
    with open(os.devnull, 'w') as FNULL:
        sp.run(['tiny2', 'temp.in'], stdout=FNULL, stderr=sp.STDOUT)
        sp.run(['tiny3', 'temp.in', 'sub=9'], stdout=FNULL, stderr=sp.STDOUT)
    if not silent:
        print(outputRoot, '00.fits generated', sep='')
    psf_fn = inFile[1].strip() + '00_psf.fits'
    # two auxiliary files generated by PSF, remove these two files to keep the
    # working directory clean
    tt3_fn = inFile[1].strip() + '.tt3'
    os.remove(os.path.join(os.getcwd(), psf_fn))
    os.remove(os.path.join(os.getcwd(), tt3_fn))
    fn = outputRoot + '00.fits'  # PSF file name
    # move the PSF to the aim directory
    os.rename(os.path.join(os.getcwd(), fn), os.path.join(outputDIR, fn))
    return inFile[1].strip() + '00.fits'  # return the file name


if __name__ == '__main__':
    pass
    # inFile = open('2mass_psf.in', 'r').readlines()
    # jitxList = np.arange(0,50,10)
    # jityList = np.arange(0,50,10)
    # disList = np.arange(2.5, 3.5, 0.2) #fix displacement at 3.0
    # for angle in [0, 1]:
    #     for dither in range(4):
    #         for jitx in jitxList:
    #             for jity in jityList:
    #                 for dis in disList:
    #                     aimDIR = os.path.join('.','PSF', 'angle_{0}_dither_{1}'.format(angle, dither))
    #                     if not os.path.exists(aimDIR): os.mkdir(aimDIR)
    #                     modFile = modifyIn(inFile, angle, dither, jitx, jity, dis)
    #                     fn = modFile[1].strip()+'00.fits'
    #                     psf_fn = modFile[1].strip()+'00_psf.fits'
    #                     tt3_fn = modFile[1].strip()+'.tt3'
    #                     out = open('temp.in', 'w')
    #                     out.writelines(modFile)
    #                     out.close()
    #                     sp.call(['tiny2', 'temp.in'])
    #                     sp.call(['tiny3', 'temp.in', 'sub=10'])
    #                     os.remove(psf_fn)
    #                     os.remove(tt3_fn)
    #                     os.rename(fn, os.path.join(aimDIR, fn))
