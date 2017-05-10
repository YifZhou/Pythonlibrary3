#! /usr/bin/env python3

"""
plot a histogram and compare to the best fit Gaussian distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from fit import gaussFit


def plotGHist(a, bins='auto', normed=False, printparams=False):
    hist, bin_edges = np.histogram(a, bins=bins, normed=normed)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    bestGaussian = gaussFit(bins, hist)
    plt.figure()
    plt.errorbar(bins, hist, ls='steps-mid')
    plt.plot(bins, bestGaussian.best_fit)
    if printparams:
        plt.text(0.7, 0.8,
                 'amplitude={0:.3f},\n center={1:.3f},\n sigma={2:.3f}'.format(
                     bestGaussian.best_values['amplitude'],
                     bestGaussian.best_values['center'],
                     bestGaussian.best_values['sigma']),
                 transform=plt.gca().transAxes,
                 ha='left')
    return plt.gca(), bestGaussian
