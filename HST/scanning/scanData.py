#! /usr/bin/env python

import os
from os import path, makedirs
from warnings import warn
import configparser
import json

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import pickle
import shelve

from random import choice
from plot import imshow

from HST.scanning.scanFile import scanFile
from HST.scanning.scanDeRamp import scanDeRamp
from HST import wfc3Dispersion
"""scanning data time seires
"""


class scanData:
    """the class for time sieries of scanning data
    """

    def __init__(self, configFN):
        """python class for assembling HST/WFC3 scanning data time series

        :param configFN: configuration file name. Configuration file
        contains all the pre-determined configuration information
        :returns: None
        """
        self.configFN = configFN
        self.conf = configparser.ConfigParser()
        self.conf.read(configFN)
        infoFile = self.conf['directory']['infoFile']
        self.info = pd.read_csv(
            infoFile, parse_dates=True, index_col='Datetime')
        self.info.sort_values('Time')
        self.info = self.info[(self.info['Filter'] == 'G141')]
        projDIR = self.conf['directory']['projDIR']
        self.dataDIR = self.conf['directory']['dataDIR']
        self.saveDIR = self.conf['directory']['saveDIR']
        calibrationDIR = self.conf['directory']['calibrationDIR']
        skyFN = self.conf['config']['skyFN']
        skyFN = path.join(calibrationDIR, skyFN)
        flatFN = self.conf['config']['flatFN']
        flatFN = path.join(calibrationDIR, flatFN)
        self.skyMask = fits.getdata(skyFN, 0)
        self.flat = fits.getdata(flatFN, 0)
        self.direct_x = self.conf['config'].getfloat('direct_x')
        self.direct_y = self.conf['config'].getfloat('direct_y')
        self.ROI = json.loads(self.conf['config']['ROI'])
        self.XOI = np.arange(self.ROI[0],
                             self.ROI[1])  # list of x index within ROI
        self.yDataStart = self.conf['config'].getint('yDataStart')
        self.yDataEnd = self.conf['config'].getint('yDataEnd')
        self.subarray = self.conf['config'].getint('subarray')
        # full wavelength range in micron
        self.wavelength = wfc3Dispersion(
            self.direct_x, self.direct_y, subarray=self.subarray) / 1e4
        self.waveOI = self.wavelength[self.XOI]  # wavelength in ROI
        self.scanFileList = []
        self.time = self.info['Time'].values
        self.orbit = self.info['Orbit'].values
        self.expTime = self.info['Exp Time'].values[0]
        self.xShift = self.info['DX'].values
        self.scanDirect = self.info['ScanDirection'].values
        self.nFrame = len(self.time)

        # read in median cubes
        medianFNs = json.loads(self.conf['config']['medianFNs'])
        medianFNs = [
            path.join(calibrationDIR, medianFN) for medianFN in medianFNs
        ]
        self.twoDirect = self.conf['config'].getboolean('twoDirect')
        if type(medianFNs) != list:
            medianFNs = [medianFNs]
        with open(medianFNs[0], 'rb') as pkl:
            self.medianCube0 = pickle.load(pkl)
        if self.twoDirect:
            try:
                with open(medianFNs[1], 'rb') as pkl:
                    self.medianCube1 = pickle.load(pkl)
            except FileNotFoundError:
                self.medianCube1 = self.medianCube0
        self.twoDirectScale = 1.0

        # extracted light curve stores in 2D arrays
        # First dimension is the x index/wavelength
        # Second dimension is time
        # Both count rate are stored
        self.LCACQUIRED = False
        self.RAMPREMOVED = False
        self.countMat = np.zeros((len(self.XOI), len(self.time)))
        self.countErrMat = np.zeros((len(self.XOI), len(self.time)))
        self.totalCountMat = np.zeros((len(self.XOI), len(self.time)))
        self.totalCountErrMat = np.zeros((len(self.XOI), len(self.time)))
        restore = self.conf['config'].getboolean('restore')
        restoreDIR = self.conf['directory']['restoreDIR']
        if restore:
            if restoreDIR is None:
                restoreDIR = self.saveDIR
            for fn in self.info['File Name']:
                print("Restoring observation {0}".format(fn))
                with open(
                        path.join(restoreDIR, fn.replace(
                            '_ima.fits', '.pickle')), 'rb') as pkf:
                    self.scanFileList.append(pickle.load(pkf))
        else:
            for i, fn in enumerate(self.info['File Name']):
                print("Processing observation {0}".format(fn))
                # use different median files for observation done in different scan direction
                if self.scanDirect[i] == 0:
                    medianCube0 = self.medianCube0
                    self.scanFileList.append(
                        scanFile(
                            fn,
                            self.dataDIR,
                            self.saveDIR,
                            self.skyMask,
                            self.flat,
                            medianCube0,
                            self.wavelength,
                            self.xShift[i],
                            self.ROI,
                            arraySize=self.subarray))

                if self.scanDirect[i] == 1:
                    medianCube1 = self.medianCube1
                    self.scanFileList.append(
                        scanFile(
                            fn,
                            self.dataDIR,
                            self.saveDIR,
                            self.skyMask,
                            self.flat,
                            medianCube1,
                            self.wavelength,
                            self.xShift[i],
                            self.ROI,
                            arraySize=self.subarray))

        # get image cube for calibration
        self.nSamp = self.scanFileList[0].nSamp  # number of samp read
        self.sampCubeList = []
        self.dqCubeList = []  # mask inclucde bad pixel and cosmic rays
        self.crCubeList = []
        for i in range(self.nSamp):
            sampCube = np.zeros((self.subarray, self.subarray, self.nFrame))
            dqCube = np.zeros((self.subarray, self.subarray, self.nFrame))
            crCube = np.zeros((self.subarray, self.subarray, self.nFrame))
            for j in range(self.nFrame):
                sampCube[:, :, j] = self.scanFileList[j].imaDataCube[:, :, i]
                dqCube[:, :, j] = self.scanFileList[j].dqCube[:, :, i]
                crCube[:, :, j] = self.scanFileList[j].crCube[:, :, i]
            self.sampCubeList.append(sampCube.copy())
            self.dqCubeList.append(dqCube.copy())
            self.crCubeList.append(crCube.copy())

    def showExampleImage(self, n=0):
        """plot one scan image. Use the last sample in the ima Array

        :param n: (default 0) the Index of the ima file
        :returns: plotted figure
        :rtype: matplotlib.figure

        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        imshow(
            self.scanFileList[n].imaDataCube[:, :, -1] -
            np.nanmin(self.scanFileList[n].imaDataCube[:, :, -1]),
            origin='lower',
            ax=ax,
            cmap='viridis')
        return fig

    def pixelLightCurve(self, x, y, plot=False):
        """get the light curve and uncertainties of a single pixel

        :param x: x coordinate Index
        :param y: y coordinate Index
        :param plot: (default False) whether to make a plot.
        :returns: light curve and the uncertainties
        :rtype: numpy array

        """

        lc = np.array([sf.pixelCount(x, y) for sf in self.scanFileList])
        lc_err = np.array([sf.pixelError(x, y) for sf in self.scanFileList])
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time, lc, yerr=lc_err, fmt='o', ls='')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.text(
                0.05,
                0.05,
                'x={0}, y={1}'.format(x, y),
                transform=ax.transAxes)
        return lc, lc_err

    def columnLightCurve(self, x, yRange, plot=False):
        """get the light curve of a column

        :param x: x index of the column
        :param yRange: y coordinate ranges to get the sum
        :param plot: (default False) whether to make a plot
        :returns: column light curve, column light curve uncertainties
        :rtype: tuple of two numpy arrays

        """
        # warn("For more precise light curve, use countLightCurve instead")
        lc = np.zeros(len(self.scanFileList))
        lc_err = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            lc[i] = sf.columnCount(x, yRange)
            lc_err[i] = sf.columnError(x, yRange)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time, lc, yerr=lc_err, fmt='o', ls='')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.text(
                0.05,
                0.05,
                'x={0}, y=({1}, {2})'.format(x, yRange[0], yRange[1]),
                transform=ax.transAxes)
        return lc, lc_err

    def countLightCurve(self, x, plot=False):
        """use totalCountSpec to get the light curve, more precise
        than columnLightCurve for better treatment of cosmic ray,
        summing region consideration

        :param x: x index of the column
        :param plot: (default False) whether to make a plot
        :returns: light curve and uncertainties with units of electrons
        :rtype: tuple of two numpy arrays

        """
        if not self.LCACQUIRED:
            print('Light curve not collected yet')
            print('collecting light curve')
            self.getLightCurve()

        lc = np.zeros(len(self.scanFileList))
        lc_err = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            lc[i] = sf.totalCountSpec[x]
            lc_err[i] = sf.totalCountSpecErr[x]
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time, lc, yerr=lc_err, fmt='o', ls='')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Total Counts')
            ax.text(0.05, 0.05, 'x={0})'.format(x), transform=ax.transAxes)
        return lc, lc_err

    def getLightCurve(self, overwrite=False):
        """Extract light curve from observation time series

        Extracted light curves and their uncertainties are saved in 2D
        numpy arrays. The four arrays are:
        countMat -- light curves in terms of average count in ROI
        countErrMat -- uncertainties for average count
        totalCountMat -- light curves in terms of total count
        totalCountErrMat -- uncertainties for total count

        :param overwrite: (default False) If LCACQUIRED flag indicates
        light curves are obtained, whether redo light curve extraction
        """
        if self.LCACQUIRED and (not overwrite):
            print("Light curve acquired already")
            print("To re-extract the light curve, use overwrite=True option")
            return

        self.LCACQUIRED = True
        for sf in self.scanFileList:
            sf.calTotalCountSpec(overwrite=overwrite)
        for i, x in enumerate(self.XOI):
            count, countErr = self.columnLightCurve(x,
                                                    [self.yDataStart, self.yDataEnd])
            totalCount, totalCountErr = self.countLightCurve(x)
            # use the mean of the ratio to determine the scale
            scale = np.mean(totalCount / count)
            self.totalCountMat[i, :] = totalCount
            self.totalCountErrMat[i, :] = totalCountErr
            self.countMat[i, :] = totalCount / scale
            self.countErrMat[i, :] = totalCountErr / scale

    def getWhiteLightCurve(self, plot=False):
        """get broadband light curve and the uncertainties

        :param plot: whether to make a plot
        :returns: broad band light curve and uncertainties
        :rtype: tuple of two numpy arrays

        """
        if not self.LCACQUIRED:
            self.getLightCurve()
        self.whiteLightCurve = self.countMat.sum(axis=0)
        self.whiteLightCurveErr = np.sqrt((self.countErrMat**2).sum(axis=0))
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(
                self.time,
                self.whiteLightCurve,
                yerr=self.whiteLightCurveErr,
                fmt='.',
                ls='',
                ms=6)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Total Counts')
            ax.set_title('White Light Curve')
        return self.whiteLightCurve, self.whiteLightCurveErr

    def removeRamp(self, initFN, overwrite=False):
        """use RECTE model to remove the ramp effect

        :param initFN: file that saves the initial values and for the ramp fit
        """
        if self.RAMPREMOVED and (not overwrite):
            print("Ramp removal DONE!")
            print("To redo ramp removal, use overwrite=True option")
        from lmfit import Parameters
        # get ramp paraeters form config files
        recteConf = configparser.ConfigParser()
        recteConf.read(initFN)
        plot = recteConf['general'].getboolean('plot')
        plotDIR = recteConf['general']['plotDIR']
        if plot and (not path.exists(plotDIR)):
            makedirs(plotDIR)
        p = Parameters()
        for pName in ['trap_pop_s', 'trap_pop_f', 'dTrap_s', 'dTrap_f']:
            p.add(
                pName,
                value=recteConf[pName].getfloat('value'),
                min=recteConf[pName].getfloat('min'),
                max=recteConf[pName].getfloat('max'),
                vary=recteConf[pName].getboolean('vary'))

        excludedOrbit = json.loads(recteConf['general']['excludedOrbit'])
        binSize = recteConf['general'].getint('binSize')
        weights = np.ones_like(self.time)
        if excludedOrbit is not None:
            if type(excludedOrbit) is not list:
                excludedOrbit = [excludedOrbit]
            for o in excludedOrbit:
                weights[self.orbit == o] = 0
        self.LCMat_deRamp = self.countMat.copy()
        self.LCErrMat_deRamp = self.countErrMat.copy()
        self.rampModelMat = np.zeros_like(self.countMat)
        self.LCMat_deRamp, self.LCErrMat_deRamp, self.rampModelMat = scanDeRamp(
            p, weights, self.expTime, self.countMat, self.countErrMat,
            self.time, self.XOI, self.twoDirect, self.scanDirect, binSize,
            plot, plotDIR)
        self.whiteLightCurve_deRamp = self.LCMat_deRamp.sum(axis=0)
        self.whiteLightCurveErr_deRamp = np.sqrt((self.LCErrMat_deRamp**2).sum(axis=0))
        self.RAMPREMOVED = True
        return self.LCMat_deRamp, self.LCErrMat_deRamp

    def skyTrend(self, plot=False):
        """calculate the sky background levels for each differenced frames

        :param plot: (default False) whether to make a plot
        :returns: time stamp of each differenced frame and the sky levels
        :rtype: tuples of two numpy arrays

        """

        t = []
        sky = []
        # use two list to collect time stampls and sky levels for
        # every differenced frames
        for i, sf in enumerate(self.scanFileList):
            t.append(self.time[i] + sf.imaSampTime[2:])
            sky.append(sf.skyValue[1:])
        t = np.concatenate(t)
        sky = np.concatenate(sky)
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t, sky)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Sky Count [$\mathsf{e^-}$]')
        return t, sky

    def plotScanRate(self):
        """plot the profiles along the scanning direction
        The scan rate plot serves as a diagnostic of the data reduction quality

        :returns: plotted figure
        :rtype: matplotlib.figure

        """

        fig, ax = plt.subplots()
        yPixels = np.arange(self.ROI[2], self.ROI[3])
        for i, sf in enumerate(self.scanFileList):
            ax.plot(
                yPixels,
                sf.scanRate,
                lw=0.5,
                ls='steps',
                color=choice(['r', 'g', 'b']),
                alpha=0.8)
        ax.set_xlabel('Y')
        ax.set_ylabel('scan rate')
        return fig

    def calcTwoDirectScale(self, nStart=3):
        """Use broad band light curves to calculate the factors of two
        scanning directions. Taking the average of the upp scanning
        light curve and down scanning light curve to get the factor

        :param nStart: number of frames to ignore at the beginning of
        the light curve
        :returns: fator
        :rtype:

        """

        wlc, wlc_err = self.whiteLC()
        scanCorrFactList = np.zeros(2)
        for i, orbit in enumerate((1, 3)):
            # remove first several exposures affected by ramp effect
            lc_upp = wlc[(self.orbit == orbit) & (self.scanDirect == 0)]
            lc_down = wlc[(self.orbit == orbit) & (self.scanDirect == 1)]
            scanCorrFactList[i] = (lc_upp[nStart:]).mean() / (
                lc_down[nStart:]).mean()
        return scanCorrFactList.mean()

    def save(self):
        """save each scanning file to pickle

        :returns: None

        """

        for sf in self.scanFileList:
            sf.saveDIR = self.saveDIR
            sf.save()


if __name__ == '__main__':
    pass
