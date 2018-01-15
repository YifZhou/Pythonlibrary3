#! /usr/bin/env python

import os
from os import path
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import pickle
import shelve


from random import choice
from plot import imshow

from .scanFile import scanFile

"""scanning data time seires
"""


class scanData(object):
    """the class for time sieries of scanning data
    """

    def __init__(self,
                 infoFile,
                 fileDIR,
                 saveDIR,
                 skyFN,
                 flatFN,
                 ROI,
                 twoDirect=False,
                 restore=False,
                 restoreDIR=None):
        """python class for assembling HST/WFC3 scanning data time series

        :param infoFile: .csv file inlucde all the file informations
        :param fileDIR: .fits file directory
        :param saveDIR: .pickle/shelve direction
        :param skyFN: pre-calculated sky Image directory
        :param flatFN: pre-calculated flat field directory
        :param ROI: region of interest. Region used for process
        :param twoDirect: (default False) whether bi-directional
        scanning mode applied
        :param restore: (default False) whether restoring previous calculated result
        :param restoreDIR: (default None) restoration directory
        :returns: None

        """

        super(scanData, self).__init__()
        self.info = pd.read_csv(
            infoFile, parse_dates=True, index_col='Datetime')
        self.info.sort_values('Time')
        self.info = self.info[(self.info['Filter'] == 'G141')]
        self.skyMask = fits.getdata(skyFN, 0)
        self.flat = fits.getdata(flatFN, 0)
        self.ROI = ROI
        self.scanFileList = []
        self.saveDIR = saveDIR
        self.time = self.info['Time'].values
        self.orbit = self.info['Orbit'].values
        self.expTime = self.info['Exp Time'].values[0]
        self.xShift = self.info['DX'].values
        # self.xShift = 0
        self.scanDirect = self.info['ScanDirection'].values
        with open('./diff_median_scan_0.pkl', 'rb') as pkl:
            self.medianDiff0 = pickle.load(pkl)
        with open('./diff_median_scan_1.pkl', 'rb') as pkl:
            self.medianDiff1 = pickle.load(pkl)
        self.twoDirectScale = 1.0

        if restore:
            if restoreDIR is None:
                restoreDIR = saveDIR
            for fn in self.info['File Name']:
                with open(
                        path.join(restoreDIR, fn.replace(
                            '_ima.fits', '.pickle')), 'rb') as pkf:
                    self.scanFileList.append(pickle.load(pkf))
        else:
            for i, fn in enumerate(self.info['File Name']):
                if self.scanDirect[i] == 0:
                    medianDiff = self.medianDiff0
                if self.scanDirect[i] == 1:
                    medianDiff = self.medianDiff1
                # fn = 'id4301ptq_ima.fits'
                # medianDiff = self.medianDiff1
                print(fn)
                self.scanFileList.append(
                    scanFile(fn, fileDIR, saveDIR, self.skyMask, self.flat,
                             medianDiff, self.xShift[i], self.ROI))

    def showExampleImage(self, n=0):
        """plot one scan image. Use the last sample in the ima Array

        :param n: (default 0) the Index of the ima file
        :returns: plotted figure
        :rtype: matplotlib.figure

        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(
            self.scanFileList[n].imaDataCube[:, :, -1] -
            np.nanmin(self.scanFileList[n].imaDataCube[:, :, -1]),
            origin='lower',
            norm=LogNorm(),
            cmap='viridis')
        fig.colorbar(cax)
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

    def whiteLC(self, plot=False):
        """get broadband light curve

        :param plot: whether to make a plot
        :returns: broad band light curve and uncertainties
        :rtype: tuple of two numpy arrays

        """

        wlc = np.zeros(len(self.scanFileList))
        wlc_err = np.zeros(len(self.scanFileList))
        for i, sf in enumerate(self.scanFileList):
            count, error = sf.white()
            wlc[i] = count
            wlc_err[i] = error
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.time, wlc, yerr=wlc_err, fmt='.', ls='', ms=6)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Counts')
            ax.set_title('White Light Curve')
        return wlc, wlc_err

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

    def save(self):
        """save each scanning file to pickle

        :returns: None

        """

        for sf in self.scanFileList:
            sf.saveDIR = self.saveDIR
            sf.save()

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


if __name__ == '__main__':
    visit = 1
    xStart = 70
    xEnd = 200
    yStart = 135
    yEnd = 220
    yDataStart = 164
    yDataEnd = 184
    rootDIR = path.expanduser('~/Documents/TRAPPIST-1')
    analysisDIR = path.join(rootDIR, 'Analysis')
    dataDIR = path.join(rootDIR, 'Data')
    infoFN = path.join(analysisDIR,
                       'TRAPPIST_visit_{0}_shiftInfo.csv'.format(visit))
    saveDIR = path.join(rootDIR, 'pickle/visit_{0}'.format(visit))
    skyFN = path.expanduser('~/Documents/TRAPPIST-1/analysis/skymask.fits')
    flatFN = path.expanduser('~/Documents/TRAPPIST-1/analysis/flatfield.fits')
    sd = scanData(
        infoFN,
        dataDIR,
        saveDIR,
        skyFN,
        flatFN, [xStart, xEnd, yStart, yEnd],
        twoDirect=True,
        restore=False,
        restoreDIR=saveDIR)
    sd.save()
    # collect light curves
    xList = list(range(xStart, xEnd))
    LCmatrix = np.zeros((len(xList), len(sd.time)))
    Errmatrix = np.zeros((len(xList), len(sd.time)))
    LCmatrix0 = np.zeros((len(xList), len(sd.time)))
    Errmatrix0 = np.zeros((len(xList), len(sd.time)))
    twoDirectScale = sd.calcTwoDirectScale()
    upIndex = np.where(sd.scanDirect == 0)[0]
    downIndex = np.where(sd.scanDirect == 1)[0]
    plt.close('all')
    wlc, wlc_err = sd.whiteLC(plot=True)
    fig = plt.gcf()
    fig.savefig(
        path.join(rootDIR, 'whitelc',
                  'whitelc_visit_{0:02d}.pdf').format(visit))
    for j, x in enumerate(xList):
        lc, lc_err = sd.columnLightCurve(x, [yDataStart, yDataEnd])
        lc0, lc_err0 = sd.countLightCurve(x)
        # scaleUp = lc0[upIndex].mean() / lc[upIndex].mean()
        # scaleDown = lc0[downIndex].mean() / lc[downIndex].mean()
        scale = lc0.mean() / lc.mean()
        scaleUp = scale
        scaleDown = scaleUp / twoDirectScale
        lc[upIndex] = lc0[upIndex] / scaleUp
        lc[downIndex] = lc0[downIndex] / scaleDown
        lc_err[upIndex] = lc_err0[upIndex] / scaleUp
        lc_err[downIndex] = lc_err0[downIndex] / scaleDown
        # lc[downIndex] = lc[downIndex] / twoDirectScale
        # scale = lc0.mean() / lc.mean()
        # lc = lc0 / scale
        # lc_err = lc_err0 / scale
        LCmatrix[j, :] = lc
        Errmatrix[j, :] = lc_err
        LCmatrix0[j, :] = lc * scale
        Errmatrix0[j, :] = lc_err * scale
    fig1, ax1 = plt.subplots()
    wlc = LCmatrix0.sum(axis=0)
    elc = np.sqrt((Errmatrix0**2).sum(axis=0))
    ax1.errorbar(sd.time, wlc, yerr=elc, ls='', fmt='.')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Flux')
    np.savetxt('whitelc.dat', np.array([sd.time, wlc, elc]).T)
    fig1.savefig(
        path.join(rootDIR, 'whitelc',
                  'whitelc0_visit_{0:02d}.pdf'.format(visit)))
    db = shelve.open(
        path.join(saveDIR, 'LCmatrix_visit_{0:02d}.shelve'.format(visit)))
    db['LCmatrix'] = LCmatrix
    db['Errmatrix'] = Errmatrix
    db['time'] = sd.time
    db['xList'] = xList
    db['orbit'] = sd.orbit
    db['expTime'] = sd.expTime
    db.close()
    db = shelve.open(
        path.join(saveDIR, 'LCmatrix_visit_{0:02d}_sky.shelve'.format(visit)))
    t, sky = sd.skyTrend()
    db['time'] = t
    db['sky'] = sky
    db.close()
