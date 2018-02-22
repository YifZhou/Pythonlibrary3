#! /usr/bin/env python
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals
from glob import glob
from os import path
import pandas as pd
from astropy.io.fits import getheader
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astro import MJD2BJD


def timeSeriesOrbit(time, maxInterval=600):
    """use expstar time to determing orbit numbers of each exposures

    :param time: time stamps of each exposures
    :param maxInterval: maximum interval between exposure in same orbits

    """
    t0 = 0
    time0 = time - time[0]
    orbit_i = 0
    orbit = np.zeros(len(time), dtype=int)
    for i in range(len(orbit)):
        dt = time0[i] - t0
        t0 = time0[i]
        if dt > maxInterval:
            orbit_i += 1
        orbit[i] = orbit_i
    return orbit


def timeSeriesInfo(fileType='flt',
                   dataDIR=None):
    """generic python function to collect informations for HST time series

    :param fileType: type of fits file
    :param dataDIR: directory that stores the files
    :returns: info dataframe
    :rtype: pandas dataframe

    """

    # targetCoord = SkyCoord(sourceCoord[0], sourceCoord[1],
    #                        unit=(u.hourangle, u.deg), frame='fk5')
    if dataDIR is None:
        dataDIR = './'
    # compile file list with correct visit number
    fnList = glob(path.join(dataDIR, '*{0}.fits'.format(fileType)))
    # get the source information from the first file
    hd0 = getheader(fnList[0], 0)
    ra0 = hd0['RA_TARG']
    dec0 = hd0['DEC_TARG']
    targetCoord = SkyCoord(ra0, dec0, unit=u.deg, frame='icrs')
    # collect informations for all files
    filterList = []
    dateList = []
    timeList = []
    exptimeList = []
    expStartMJDList = []
    expFlagList = []
    SAATimeList = []
    for i, fn in enumerate(fnList):
        hd = getheader(fn, 0)
        filterList.append(hd['FILTER'])
        dateList.append(hd['date-obs'])
        timeList.append(hd['time-obs'])
        exptimeList.append(hd['exptime'])
        expStartMJDList.append(hd['expstart'])
        expFlagList.append(hd['expflag'])
        SAATimeList.append(hd['saa_time'])
        # orbitNoList = orbitNo(fnList)

    dtList = pd.to_datetime(
        [' '.join([date, time]) for date, time in zip(dateList, timeList)])
    df = pd.DataFrame()
    df['File Name'] = [path.basename(fn) for fn in fnList]
    df['Datetime'] = dtList
    df['Exp Time'] = exptimeList
    df['Filter'] = filterList
    df['JD'] = [expStart + 2400000.5 for expStart in expStartMJDList]
    df['BJD_TDB'] = [
        MJD2BJD(expStart, targetCoord) for expStart in expStartMJDList
    ]
    df['EXPFLAG'] = expFlagList
    df['SAA'] = SAATimeList

    # df['Orbit'] = orbitNoList
    df.set_index('Datetime')
    df = df.sort_values('Datetime')
    df['Time'] = (df['JD'] - df['JD'].values[0]) * 86400
    df['Orbit'] = timeSeriesOrbit(df['Time'].values)
    return df


if __name__ == '__main__':
    Target = 'TRAPPIST'
    rootDIR = path.expanduser('~/Documents/TRAPPIST-1/')
    dataDIR = path.join(rootDIR, 'Data')
    analysisDIR = path.join(rootDIR, 'analysis')
    timeSeriesInfo(Target, 1, rootDIR, dataDIR, analysisDIR)
