#! /usr/bin/env python
import os
import pickle
from astropy.io import fits
from os import path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.ndimage.interpolation import shift

import HST
from HST import badPixelInterp, dqMask
from astro import cosmics
from plot import imshow
"""python class to read in ima files
"""


class scanFile:
    def __init__(self,
                 fileName,
                 fileDIR,
                 saveDIR,
                 skyMask,
                 flat,
                 medianCube,
                 wavelength,
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
        :param medianCube: pre-calculated median differenced frames
        :param wavelength: wavelength solution for the observation
        :param xshift: shift in x-direction
        :param ROI: Region of interest. ROI defines the region of interest for cosmic Ray removal
        :param arraySize: (default 256) subarray size

        """
        # set FLAGS first
        self.CRFOUND = False
        self.INTERPOLATED = False
        self.SKYREMOVED = False
        self.COUNTCALCULATED = False
        self.skyMask = skyMask.copy()
        self.flat = flat
        self.fileDIR = fileDIR
        self.saveDIR = saveDIR
        self.xshift = xshift
        self.wavelength = wavelength
        self.ROI = ROI
        self.arraySize = arraySize
        self.imaFN = path.join(fileDIR, fileName)
        self.xList = np.arange(ROI[0], ROI[1])
        imaHeader = fits.getheader(self.imaFN, 0)
        self.rootName = imaHeader['ROOTNAME']
        self.nSamp = imaHeader['NSAMP']
        self.expTime = imaHeader['EXPTIME']
        # unit: counts, specifically for scanning data file
        self.imaDataCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.imaDataFixedCube = np.ma.MaskedArray(
            data=np.zeros(self.imaDataCube.shape),
            mask=np.zeros(self.imaDataCube.shape))
        self.errCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.dqCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.crCube = np.zeros((arraySize, arraySize, self.nSamp))
        self.imaSampTime = np.zeros(self.nSamp)
        # unit: counts
        self.imaDataDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.errDiff = np.zeros((arraySize, arraySize, self.nSamp - 1))
        self.skyVar = np.zeros(self.nSamp - 1)  # error of sky subtraction
        self.medianCube = medianCube
        self.readIma()

    def readIma(self, fixBadPixel=True):
        """Read ima file. sci, err, and dq arrays are saved in
        imaDataCube, errCube and dqCube. ima files save the array
        backawardly. However, for the cubes, they save data ascending
        with time. The difference of the sci frames are saved in imaDiffCube

        :param fixBadPixel: whether to interpolate the bad pixels
        :returns: None

        """
        print("Reading file: {0}".format(path.basename(self.imaFN)))
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
            if fixBadPixel:
                self.findCosmics(overwrite=True)
                self.interpBadPixels(overwrite=True)
            else:
                # if do not fix bad pixel, then just copy imaDataCube, and set mask to be zero
                for i in range(self.nSamp):
                    self.CRFOUND = False
                    self.INTERPOLATED = False
                    self.imaDataFixedCube.data[:, :,
                                               i] = self.imaDataCube[:, :, i]
                    self.imaDataFixedCube.mask[:, :, i] = np.zeros_like(
                        self.dqCube[:, :, i])
            self.calDiffCube()
            self.removeSky()

    def findCosmics(self, sigma=3, overwrite=False):
        """recognize cosmic rays in each frames, generate cosmic ray mas and save it in crMask

        :param sigma: (default 3) threshold for cosmic ray recognition
        :param overwrite: (default False) if cosmic recognition is done, whether redo and overwrite the result
        :returns: None
        :rtype: None

        """
        # check if CR recoginition is done
        try:
            done = self.CRFOUND
        except NameError:
            done = False
        if done and (not overwrite):
            print("Cosmic ray recognition is done.")
            print(
                "Use option 'overwrite=True' to redo and overwrite the result")
        else:
            # intialize the list that saves the cosmic ray results
            self.crList = []
            for i in range(self.nSamp):
                crMask, xCR, yCR = cosmics(
                    self.imaDataCube[:, :, i],
                    self.errCube[:, :, i],
                    self.dqCube[:, :, i],
                    self.medianCube[:, :, i],
                    sigma=3)
                self.crList.append(list(zip(xCR, yCR)))
                self.crCube[:, :, i] = crMask
        self.CRFOUND = True

    def interpBadPixels(self,
                        fixDQ=True,
                        fixCR=True,
                        method='linear',
                        overwrite=False,
                        updateDiffCube=False):
        """Use simple linear interpolation to fix bad pixel

        :param fixDQ: (default True) whether to interpolate pixel with data quality flag
        :param fixCR: (default True) whether to interpolate pixel that is marked as cosmic ray affected pixel
        :param method: (default 'linear') interpolation method, options are 'linear' and 'cubic
        :param overwrite: (default False) if pixel interpolation is done, whether redo and overwrite the result
        :param updateDiffCube: (default False) whether to update the differential array

        """
        try:
            done = self.INTERPOLATED
        except NameError:
            done = False
            self.INTERPOLATED = False
        if done and (not overwrite):
            print("Cosmic ray recognition is done.")
            print(
                "Use option 'overwrite=True' to redo and overwrite the result")
            return

        for i in range(self.nSamp):
            mask = np.zeros_like(self.dqCube[:, :, 0])
            if fixDQ:
                mask += self.dqCube[:, :, i]
            if fixCR:
                mask += self.crCube[:, :, i]
            goody, goodx = np.where(mask == 0)
            bady, badx = np.where(mask > 0)
            fixedIm = self.imaDataCube[:, :, i].copy()
            interpv = griddata(
                (goody, goodx),
                self.imaDataCube[goody, goodx, i], (bady, badx),
                method='linear')
            fixedIm[bady, badx] = interpv
            # save the result to self.imaDataFixedCube
            self.imaDataFixedCube.data[:, :, i] = fixedIm
            self.imaDataFixedCube.mask[:, :, i] = mask
        self.INTERPOLATED = True
        if updateDiffCube:
            self.calDiffCube()

    def calDiffCube(self):
        """calculate the differential array and its error array
        """

        for i in range(self.nSamp - 1):
            self.imaDataDiff[:, :, i] =\
                                        (self.imaDataFixedCube.data[:, :, i+1] -
                                         self.imaDataFixedCube.data[:, :, i])
            mask = self.dqCube[:, :, i] + self.crCube[:, :, i]
            goody, goodx = np.where(mask == 0)
            bady, badx = np.where(mask > 0)
            fixedIm = self.imaDataDiff[:, :, i].copy()
            interpv = griddata(
                (goody, goodx),
                self.imaDataDiff[goody, goodx, i], (bady, badx),
                method='linear')
            fixedIm[bady, badx] = interpv
            self.imaDataDiff[:, :, i] = fixedIm

            # FIXME: Nov 15: parameter errDiff is not used, why bother?
            self.errDiff[:, :, i] = self.errCube[:, :, i + 1]**2

    def removeSky(self, klipthresh=5, overwrite=False):
        """apply sky subtraction on differenced frame. It is
        more accurate to remove sky in *differenced* frames (Deming et al. 2013)

        :param klipthresh: (default 5) threshold in sigma_clip
        algorithms
        :param overwrite: if sky is removed, whether to calculat it again
        :returns: None
        """
        if (self.SKYREMOVED is True) and (not overwrite):
            print("Kky removal recognition is done.")
            print(
                "Use option 'overwrite=True' to redo and overwrite the result")
            return

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

    def calTotalCountSpec(self, radius=20, nSampStart=0, overwrite=False):
        """calculate column sums for every column

        :param radius: number of pixels at each side of the peak of
        the scanning region
        :param nSampStart: which ima frame to start
        :param overwrite: (default False) if COUNTCALCULATED indicate this function is done, whether re-do it
        :returns: count and uncertanties
        :rtype: tuple of two float

        """
        if self.COUNTCALCULATED and (not overwrite):
            print("total count is calculated.")
            print(
                "Use option 'overwrite=True' to redo and overwrite the result")
            return self.totalCountSpec, self.totalCountSpecErr
        nFrame = self.nSamp - 1 - nSampStart
        self.totalCountSpec = np.zeros(self.arraySize)
        totalCountSpecVar = np.zeros(self.arraySize)
        # add two more parameter to save the result for cosmic ray detection
        for i in range(nSampStart, nSampStart + nFrame):
            im = self.imaDataDiff[:, :, i]
            err = np.sqrt(np.abs(im) + self.errCube[:, :, 0]**2)
            scanRate = np.nanmean(im, axis=1)
            peakID = np.argmax(scanRate)
            im_interp = im[peakID - radius:peakID + radius, :]
            err = np.sqrt(
                np.abs(im_interp) +
                self.errCube[peakID - radius:peakID + radius, :, 0]**2)
            # shift the image in x direction to make correction for wavelength and ramp effect
            im_interp = shift(im_interp, [0, self.xshift], order=1)
            err = shift(err, [0, self.xshift], order=1)
            self.totalCountSpec += np.nansum(im_interp, axis=0)
            totalCountSpecVar += np.nansum(err**2, axis=0) + self.skyVar[i]
        self.totalCountSpecErr = np.sqrt(totalCountSpecVar)
        self.COUNTCALCULATED = True
        return self.totalCountSpec, self.totalCountSpecErr

    def white(self):
        """return white count and errors
        """
        return np.sum(self.totalCountSpec[self.ROI[0]:self.ROI[1]]), \
            np.sqrt(np.sum(self.totalCountSpecErr[self.ROI[0]:self.ROI[1]]**2))

    def stellarSpectrum(self, wmin=1.1, wmax=1.7):
        """get the host star spectrum

        :param wmin: minimum boundary of wavelength
        :param wmax: maximum boundary of wavelength
        :returns: wavelength, spec, spec_err
        :rtype: tuple of numpy array

        """
        count, err = self.calTotalCountSpec()
        wid = np.where((self.wavelength > wmin) & (self.wavelength < wmax))[0]
        count = count[wid]
        err = err[wid]
        wavelength = self.wavelength[wid]
        # read the sensitivity file
        sensPath = path.join(HST.__path__[0], 'CONF/WFC3.IR.G141.1st.sens.2.fits')
        sens1st = fits.getdata(sensPath, 1)
        sensInterp = interp1d(
            sens1st['WAVELENGTH'] / 1e4, sens1st['SENSITIVITY'], kind='cubic')
        sensErrorInterp = interp1d(
            sens1st['WAVELENGTH'] / 1e4, sens1st['ERROR'], kind='cubic')
        sensAim = sensInterp(wavelength)  # map sensitivity to aimed wavelength
        sensErrAim = sensErrorInterp(wavelength)
        flux = count / sensAim / self.expTime
        fluxErr = flux * np.sqrt((err/count)**2 + (sensErrAim / sensAim)**2)
        return wavelength, flux, fluxErr

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

    def showCR(self, nSamp, xCR, yCR, imSize=8, vmin=None, vmax=None):
        """show example of recognized cosmic rays, aiming to avoid false positive

        :param nSamp: show cosmic rays from nth sample
        :param xCR: x coordinate
        :param yCR: y coordinate
        :param imSize: (default 8) the half size of the showed images
        :param vmin: (default None) minimum value range for imshow
        :param vmax: (default None) maximum value range for imshow
        :returns: plot figure
        :rtype: matplitlib.figure

        """
        if not self.crCube[yCR, xCR, nSamp]:
            print(
                "Specifed coordinate x={0}, y={1} is not recognized as cosmic ray affected, making the plot anyway".format(xCR, yCR))

        xmin = max(0, xCR - 8)
        xmax = min(255, xCR + 8)
        ymin = max(0, yCR - 8)
        ymax = min(255, yCR + 8)
        subIm = self.imaDataCube[ymin:ymax, xmin:xmax, nSamp]
        subIm_fixed = self.imaDataFixedCube.data[ymin:ymax, xmin:xmax, nSamp]
        fig = plt.figure(figsize=(8, 4))
        ax0 = fig.add_subplot(131)
        imshow(subIm, vmin=vmin, vmax=vmax, ax=ax0)
        ax0.set_title('Original Image', fontsize='small')
        ax = fig.add_subplot(132, sharex=ax0, sharey=ax0)
        imshow(subIm_fixed, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title('CR Fixed Image', fontsize='small')
        ax = fig.add_subplot(133, sharex=ax0, sharey=ax0)
        imshow(subIm-subIm_fixed, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title('Original - CR Fixed', fontsize='small')
        fig.tight_layout()
        return fig

    def showCRSlice(self, nSamp, xCR, yCR, vmin=None, vmax=None):
        """show slices around recogized cosmic ray pixel

        :param nSamp: show cosmic rays from nth sample
        :param xCR: x coordinate
        :param yCR: y coordinate
        :returns: plot figure
        :rtype: matplitlib.figure

        """
        im = self.imaDataCube[:, :, nSamp]
        err = self.errCube[:, :, nSamp]
        im_fixed = self.imaDataFixedCube.data[:, :, nSamp]
        cr = self.crCube[:, :, nSamp]
        if not cr[yCR, xCR]:
            print(
                "Specifed coordinate x={0}, y={1} is not recognized as cosmic ray affected, making the plot anyway".format(xCR, yCR))
        fig1 = plt.figure(figsize=(8, 4))
        ax1 = fig1.add_subplot(131)
        x_axis = np.arange(yCR-10, yCR+10)
        ax1.errorbar(
            x_axis,
            im[yCR - 10:yCR + 10, xCR - 1],
            yerr=err[yCR - 10:yCR + 10, xCR - 1])
        ax1.errorbar(
            x_axis,
            im_fixed[yCR - 10:yCR + 10, xCR - 1],
            yerr=err[yCR - 10:yCR + 10, xCR - 1])
        ax1.set_title('x={0}'.format(xCR - 1))

        ax2 = fig1.add_subplot(132)
        ax2.errorbar(
            x_axis,
            im[yCR - 10:yCR + 10, xCR],
            yerr=err[yCR - 10:yCR + 10, xCR])
        ax2.errorbar(
            x_axis,
            im_fixed[yCR - 10:yCR + 10, xCR],
            yerr=err[yCR - 10:yCR + 10, xCR])
        ax2.set_title('x={0}'.format(xCR))

        ax3 = fig1.add_subplot(133)
        ax3.errorbar(
            x_axis,
            im[yCR - 10:yCR + 10, xCR + 1],
            yerr=err[yCR - 10:yCR + 10, xCR + 1])
        ax3.errorbar(
            x_axis,
            im_fixed[yCR - 10:yCR + 10, xCR + 1],
            yerr=err[yCR - 10:yCR + 10, xCR + 1])
        ax3.set_title('x={0}'.format(xCR + 1))
        for ax in fig1.axes:
            ax.set_ylim([vmin, vmax])
            ax.set_xlabel('Y [pixel]')
            ax.set_ylabel('Value')
        fig1.tight_layout()

        # y slice
        fig2 = plt.figure(figsize=(8, 4))
        ax1 = fig2.add_subplot(131)
        x_axis = np.arange(xCR-10, xCR+10)
        ax1.errorbar(
            x_axis,
            im[yCR - 1, xCR - 10:xCR + 10],
            yerr=err[yCR - 1, xCR - 10:xCR + 10])
        ax1.errorbar(
            x_axis,
            im_fixed[yCR - 1, xCR - 10:xCR + 10],
            yerr=err[yCR - 1, xCR - 10:xCR + 10])
        ax1.set_title('y={0}'.format(yCR - 1))

        ax2 = fig2.add_subplot(132)
        ax2.errorbar(
            x_axis,
            im[yCR, xCR - 10:xCR + 10],
            yerr=err[yCR, xCR - 10:xCR + 10])
        ax2.errorbar(
            x_axis,
            im_fixed[yCR, xCR - 10:xCR + 10],
            yerr=err[yCR, xCR - 10:xCR + 10])
        ax2.set_title('y={0}'.format(yCR))

        ax3 = fig2.add_subplot(133)
        ax3.errorbar(
            x_axis,
            im[yCR + 1, xCR - 10:xCR + 10],
            yerr=err[yCR + 1, xCR - 10:xCR + 10])
        ax3.errorbar(
            x_axis,
            im_fixed[yCR + 1, xCR - 10:xCR + 10],
            yerr=err[yCR + 1, xCR - 10:xCR + 10])
        ax3.set_title('y={0}'.format(yCR + 1))
        for ax in fig2.axes:
            ax.set_ylim([vmin, vmax])
            ax.set_xlabel('X [pixel]')
            ax.set_ylabel('Value')
        fig2.tight_layout()
        return fig1, fig2

    def save(self, rootName=None):
        if rootName is None:
            rootName = self.rootName
        if not os.path.exists(self.saveDIR):
            os.makedirs(self.saveDIR)
        saveFN = path.join(self.saveDIR, rootName + '.pickle')
        with open(saveFN, 'wb') as pkf:
            pickle.dump(self, pkf, pickle.HIGHEST_PROTOCOL)
