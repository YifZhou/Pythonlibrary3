# -*- coding: utf-8 -*-
# /usr/bin/env python


import matplotlib.pyplot as plt
import corner
import numpy as np
"""plot mcmc fit result
"""


def plotMCMC(x, y, yerr, chain, params, func, *funcArgs,
             argName=None, nSample=50):
    """plot the mcmc fit result and compare it to the original result

    :param x: x axis variable
    :param y: y measurement
    :param yerr: y uncertainty
    :param chain: posterior chain, burnin removed
    :param params: best fit parameters
    :param func: the function to fit
    :param funcArgs: other parameters required by fit function
    :param nSample: number of sample in posterior chain
    :returns: figure
    :rtype: matplotlib.figure

    """
    nDim = chain.shape[1]
    if argName is None:
        # default arg label is empty
        argName = [''] * nDim
    fig1 = corner.corner(chain, labels=argName, smooth=1)

    fig2, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, ls='none')
    x0 = np.linspace(x.min(), x.max(), 10 * len(x))  # 10x sample
    # rate of original x
    ax.plot(x0, func(params, x0, *funcArgs), color='C3')
    sample_index = np.random.randint(chain.shape[0], size=nSample)
    for i in range(nSample):
        params_i = chain[sample_index[i], :]
        ax.plot(x0, func(params_i, x0, *funcArgs), lw=0.2,
    color='0.6')
    return fig1, fig2
