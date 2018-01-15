#! /usr/bin/env python
import os
import pickle
from astropy.io import fits
from os import path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import shift

from HST import badPixelInterp, dqMask
from astro import cosmics


"""python class to read in ima files
"""


class scanFile(object):
    def __init__(self,
                 fileName,
                 fileDIR,
                 saveDIR,
                 skyMask,
                 flat,
                 medianDiff,
                 xshift=0,
                 ROI=[0, 256, 0, 256],
                 arraySize=256):
        """python class for one ima scanning data frame

        :param fileName: fits info file name
        :param fileDIR: .fits file direction
        :param saveDIR: .pickle file direction
        :param skyMask: pre-difined mask to exclude pixels from sky
        background calculations
        :param flat: pre-calculated flat field correcton
        :param medianDiff: pre-calculated median differenced frames
        :param xshift: shift in x-direction
        :param ROI: Region of interest. ROI defines the region of interest for cosmic Ray removal
        :param arraySize: (default 256) subarray size

        """
        super(scanFile, self).__init__()
        self.skyMask = skyMask.copy()
        self.flat = flat
        self.fileDIR = fileDIR
        self.saveDIR = saveDIR
        self.xshift = xshift
        self.ROI = ROI
        self.arraySize = 256
        self.imaFN = path.join(fileDIR, fileName)
        self.xList = np.arange(ROI[0], ROI[1])
        imaHeader = fits.getheader(self.imaFN, 0)
        self.rootName = imaHeader['ROOTNAME']
        self.nSamp = imaHeader['NSAMP']
        self.expTime = imaHeader['EXPTIME']
        # unit: counts, specifically for scanning data file
        self.imaDataCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.errCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.dqCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.imaSampTime = np.zeros(self.nSamp)
        # unit: counts
        self.imaDataDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.errDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.skyVar = np.zeros(self.nSamp - 1)  # error of sky subtraction
        self.medianDiff = medianDiff
        self.readIma()
        self.totalCountSpec()

    def readIma(self):
        """Read ima file. sci, err, and dq arrays are saved in
        imaDataCube, errCube and dqCube. ima files save the array
        backawardly. However, for the cubes, they save data ascending
        with time. The difference of the sci frames are saved in imaDiffCube

        :returns: None
        """

        with fits.open(self.imaFN) as f:
            for i in range(self.nSamp):
                # ima file is stored backwardly
                # in my data, they are saved in a normal direction
                # so reverse the order for ima files
                self.imaDataCube[:, :, i] = f['sci', self.nSamp-i].\
                                            data[5:5+self.arraySize,
                                                 5:5+self.arraySize] / self.flat
                self.errCube[:, :, i] = f['err', self.nSamp-i].\
                                            data[5:5+self.arraySize,
                                                 5:5+self.arraySize]
                self.dqCube[:, :, i] = dqMask(f['dq', self.nSamp - i].data[
                    5:5 + self.arraySize, 5:5 + self.arraySize])
                self.imaSampTime[i] = f['sci', self.nSamp - i].header[
                    'SAMPTIME']
            for i in range(1, self.nSamp):
                self.imaDataCube[:, :, i] = self.imaDataCube[:, :, i] + \
                                            self.imaDataCube[:, :, 0]
            for i in range(self.nSamp - 1):
                self.imaDataDiff[:, :, i] =\
                    (self.imaDataCube[:, :, i+1] - self.imaDataCube[:, :, i])
                # Nov 15: parameter errDiff is not used, why bother?
                self.errDiff[:, :, i] = self.errCube[:, :, i + 1]**2
            self.removeSky()

    def removeSky(self, klipthresh=5):
        """apply sky subtraction on differenced frame. It is
        more accurate to remove sky in *differenced* frames (Deming et al. 2013)

        :param klipthresh: (default 5) threshold in sigma_clip
        algorithms
        :returns: None
        """
        self.skyValue = np.ones(self.nSamp - 1)
        for i in range(self.nSamp - 1):
            skymasked = np.ma.masked_array(
                self.imaDataDiff[:, :, i], mask=self.skyMask)
            # 5-iteration sigma-clip method
            skymasked = sigma_clip(skymasked, sigma=klipthresh, iters=5)
            self.skyValue[i] = np.ma.median(skymasked)
            self.skyVar[i] = np.ma.std(skymasked)**2
            self.imaDataDiff[:, :, i] = self.imaDataDiff[:, :, i] -\
                self.skyValue[i]

    def pixelLightCurve(self, x, y):
        """return the sequence of ima read

        :param x: x coordinate
        :param y: y coordinate
        :returns:  ima read sequence
        :rtype: np.array

        """
        return self.imaDataCube[y, x, :]

    def pixelDiffLightCurve(self, x, y):
        """return the sequence of the difference in ima frames

        :param x: x coordinate
        :param y: y coordinate
        :returns: ima read sequence
        :rtype: np.array

        """

        return self.imaDataDiff[y, x, :]

    def pixelCount(self, x, y, nSampStart=1):
        """total count for specific pixels, the count from 0 to 1st read by
        default is discarded by default

        :param x: x coordinate
        :param y: y coordinate
        :param nSampStart: starting samp number
        :returns: pixel counts
        :rtype: float
        """

        return np.nansum(self.imaDataDiff[y, x, nSampStart:])

    def pixelError(self, x, y):
        """
        return the error for a pixel
        combinations of the error in the last frame and sky background
        """
        return np.sqrt(self.errCube[y, x, -1]**2 + self.skyVar.sum())

    def columnCount(self, x, yRange, nSampStart=1):
        """caculate the average count of a column

        :param x: x axis of the column
        :param yRange: range of the column
        :param nSampStart: which frame to start
        :returns: total count of the specified column
        :rtype: float

        """

        column = np.array([
            self.pixelCount(x, y, nSampStart)
            for y in range(yRange[0], yRange[1])
        ])
        return np.nanmean(column)

    def columnError(self, x, yRange, nSampStart=1):
        """caculate the uncertainties average count of a column

        :param x: x axis of the column
        :param yRange: range of the column
        :param nSampStart: which frame to start
        :returns: total count of the specified column
        :rtype: float

        """
        error = np.array(
            [self.pixelError(x, y) for y in range(yRange[0], yRange[1])])
        return np.sqrt(
            np.nanmean(error**2) / (len(error) - np.isnan(error).sum()))

    def totalCountSpec(self,
                       radius=20,
                       nSampStart=0):
        """calculate column sums for every column

        :param radius: number of pixels at each side of the peak of
        the scanning region
        :param nSampStart: which ima frame to start
        :returns: count and uncertanties
        :rtype: tuple of two float

        """

        nFrame = self.nSamp - 1 - nSampStart
        self.totalCountSpec = np.zeros(self.arraySize)
        totalCountSpecVar = np.zeros(self.arraySize)
        # add two more parameter to save the result for cosmic ray detection
        self.imaDataCRRemoved = np.ma.MaskedArray(
            data=np.zeros(self.imaDataDiff.shape),
            mask=np.zeros(self.imaDataDiff.shape))
        CRList = []
        # find cosmic rays and calculate the sum
        for i in range(nSampStart, nSampStart + nFrame):
            im = self.imaDataDiff[:, :, i]
            err = np.sqrt(np.abs(im) + self.errCube[:, :, 0]**2)
            crMask, xCR, yCR = cosmics(im, err, self.dqCube[:, :, i],
                                       self.medianDiff[:, :, i])
            for yCR_i, xCR_i in zip(yCR, xCR):
                CRList.append((yCR_i, xCR_i, i))
            nanMask = np.zeros_like(crMask)
            nanMask[np.where(np.isnan(im))] = 1
            mask = (nanMask + crMask + self.dqCube[:, :, i]).astype(bool)
            scanRate = np.ma.mean(np.ma.masked_array(im, mask=mask), axis=1)
            peakID = np.argmax(scanRate)
            im_interp = badPixelInterp(im, mask)
            self.imaDataCRRemoved.data[:, :, i] = im_interp
            self.imaDataCRRemoved.mask[:, :, i] = crMask
            im_interp = im_interp[peakID - radius:peakID + radius, :]
            err = np.sqrt(
                np.abs(im_interp) +
                self.errCube[peakID - radius:peakID + radius, :, 0]**2)
            # shift the image in x direction to make correction for wavelength and ramp effect
            im_interp = shift(im_interp, [0, self.xshift], order=1)
            err = shift(err, [0, self.xshift], order=1)
            self.totalCountSpec += np.nansum(im_interp, axis=0)
            totalCountSpecVar += np.nansum(err**2, axis=0) + self.skyVar[i]
        # put CRList into a dataframe for better access
        nCR = len(CRList)
        print("{0} cosmic ray corrected for file".format(nCR) + self.imaFN)
        # save cosmic ray results
        self.CRList = pd.DataFrame(
            0, index=np.arange(nCR), columns=['nSamp No.', 'xCR', 'yCR'])
        for i in range(len(CRList)):
            self.CRList['nSamp No.'].values[i] = CRList[i][2]
            self.CRList['xCR'].values[i] = CRList[i][1]
            self.CRList['yCR'].values[i] = CRList[i][0]
        self.totalCountSpecErr = np.sqrt(totalCountSpecVar)
        return self.totalCountSpec, self.totalCountSpecErr

    def white(self):
        """return white count and errors
        """
        return np.sum(self.totalCountSpec[self.ROI[0]:self.ROI[1]]), \
            np.sqrt(np.sum(self.totalCountSpecErr[self.ROI[0]:self.ROI[1]]**2))

    def plotSampleImage(self, nSamp):
        """plot one example image from imaDataCube

        :param nSamp: sample Index
        :returns: figure with plot
        :rtype: matplotlib.figure

        """

        if nSamp >= self.nSamp:
            print('Maximum Sample number is {0}'.format(self.nSamp - 1))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(
                self.imaDataCube[:, :, nSamp],
                vmin=-200,
                vmax=200,
                origin='lower')
            fig.colorbar(cax)
            ax.text(
                0.02,
                0.02,
                'Sample: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp),
                transform=ax.transAxes,
                backgroundcolor='0.9')
            ax.text(
                0.02,
                0.08,
                'Sample Time: {0:.2f}'.format(self.imaSampTime[nSamp]),
                transform=ax.transAxes,
                backgroundcolor='0.9')
        return fig

    def plotDiffImage(self, nSamp):
        """plot one difference image from imaDataCube

        :param nSamp: sample Index
        :returns: figure with plot
        :rtype: matplotlb.figure

        """
        if nSamp >= self.nSamp:
            print('Maximum Diff Sample number is {0}'.format(self.nSamp - 2))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(
                self.imaDataDiff[:, :, nSamp],
                vmin=-200,
                vmax=200,
                origin='lower')
            fig.colorbar(cax)
            ax.text(
                0.02,
                0.02,
                'Diff No.: {0:d}/{1:d}'.format(nSamp + 1, self.nSamp - 1),
                transform=ax.transAxes,
                backgroundcolor='0.9')
            ax.text(
                0.02,
                0.08,
                'Diff Samp Time: {0:.2f}-{1:.2f}'.format(
                    self.imaSampTime[nSamp + 1], self.imaSampTime[nSamp]),
                transform=ax.transAxes,
                backgroundcolor='0.9')
        return fig

    def save(self, rootName=None):
        if rootName is None:
            rootName = self.rootName
        if not os.path.exists(self.saveDIR):
            os.makedirs(self.saveDIR)
        saveFN = path.join(self.saveDIR, rootName + '.pickle')
        with open(saveFN, 'wb') as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)
