#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Interpolation using Gaussian Process Regression (kriging).

Uses the GP pacakge from 'sklearn' to interpolate spatial data points (2d
fields).

"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.ndimage.interpolation import shift
from scipy.linalg import cholesky, cho_solve, solve_triangular
from image import rebin
import emcee
import corner
np.random.seed(1)


class Kriging2d(object):
    """Interpolate using Gaussian Process Regression (kriging).

    Uses the GP pacakge from 'sklearn' to interpolate spatial data points (2d
    fields).
    """

    def __init__(self, length_scale, amp=1.0):
        k = amp * RBF(length_scale=length_scale)
        self.gp = GaussianProcessRegressor(
            kernel=k, normalize_y=True, n_restarts_optimizer=20)

    def kriging(self, X, y):
        """Interpolate using Gaussian Process Regression (kriging).

        Uses the GP pacakge from 'sklearn' to interpolate spatial data points
        (2d).

        Interpolation equal noiseless case, i.e., "almost" no uncertainty in
        the observations.

        Bounds are defined assuming anisotropy.

        """
        # instanciate a Gaussian Process model

        # fit to data using Maximum Likelihood Estimation of the parameters
        self.gp.fit(X, y)

        # evaluate the prediction points (ask for MSE as well)

        # return [y_pred, np.sqrt(MSE)]
        return self.gp

    def kriging_mcmc(self,
                     X,
                     y,
                     nWalkers=64,
                     nSteps=500,
                     burnin=300,
                     plot=False):
        """using mcmc to find the best fitted hyper parameters"""
        self.gp.fit(X, y)

        # def lnlike(theta):
        #     return self.gp.log_marginal_likelihood(np.log(params))

        params0 = self.gp.kernel_.theta
        nDim = len(params0)
        pos0 = [
            params0 * (1 + 1e-5 * np.random.randn(nDim))
            for j in range(nWalkers)
        ]
        sampler = emcee.EnsembleSampler(
            nWalkers, nDim, self.gp.log_marginal_likelihood, threads=4)
        sampler.run_mcmc(pos0, nSteps)
        self.best_params = np.zeros(nDim)
        self.params_error = np.zeros(nDim)
        self.best_params = np.exp(
            np.median(sampler.flatchain[burnin * nWalkers:, :], axis=0))
        self.params_error = np.exp(
            sampler.flatchain[burnin * nWalkers:, :]).std(axis=0)
        if plot:
            corner.corner(
                np.exp(sampler.flatchain[burnin * nWalkers:, ]),
                quantiles=[0.25, 0.5, 0.75])
        self.gp.kernel_.theta = np.log(self.best_params)
        K = self.gp.kernel_(self.gp.X_train_)
        K[np.diag_indices_from(K)] += self.gp.alpha
        self.gp.L_ = cholesky(K, lower=True)  # Line 2
        self.gp.alpha_ = cho_solve((self.gp.L_, True), self.gp.y_train_)  # Line 3

    def kriging_fix(self, X, y, theta):
        """fix the kriging parameter"""
        self.gp.n_restarts_optimizer = 0  # avoid restart
        self.gp.fit(X, y)
        self.gp.kernel_.theta = np.log(theta)
        K = self.gp.kernel_(self.gp.X_train_)
        K[np.diag_indices_from(K)] += self.gp.alpha
        self.gp.L_ = cholesky(K, lower=True)
        self.gp.alpha_ = cho_solve((self.gp.L_, True), self.gp.y_train_)
        self.gp.n_restarts_optimizer = 20  # set the restart optimizer back to 20

    def interpolate(self, Xpred, shape=None):
        if shape is None:
            shape = (Xpred.shape[0], )
        Ypred = self.gp.predict(Xpred)
        return Ypred.reshape(shape)


if __name__ == '__main__':
    # start the testing with simple Gaussian functions
    from functions import gaussian2d
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.close('all')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    x0 = 5
    y0 = 5
    amp = 1.0
    sigma_x = 0.8
    sigma_y = 0.8
    superSamp = 10
    xx0, yy0 = np.mgrid[0:2 * x0:1 / superSamp, 0:2 * y0:1 / superSamp]
    g0 = gaussian2d(xx0, yy0, amp, x0, y0, sigma_x, sigma_y)
    # ax.plot_surface(xx0, yy0, g0, lw=0, cmap='viridis')
    g = rebin(g0, superSamp, np.sum)
    xx = rebin(xx0, superSamp, np.mean)
    yy = rebin(yy0, superSamp, np.mean)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, yy, g, lw=0, cmap='viridis')
    x_shift = .2  #
    y_shift = 0.5
    g_shift = rebin(g0[int(x_shift * superSamp):, int(y_shift * superSamp):],
                    superSamp, np.sum)
    xx_shift = rebin(xx0[int(x_shift * superSamp):, int(y_shift * superSamp):],
                     superSamp, np.mean)
    yy_shift = rebin(yy0[int(x_shift * superSamp):, int(y_shift * superSamp):],
                     superSamp, np.mean)
    x_shift1 = 0.2
    y_shift1 = 0.1
    g_shift1 = rebin(
        g0[int(x_shift1 * superSamp):, int(y_shift1 * superSamp):], superSamp,
        np.sum)
    xx_shift1 = rebin(
        xx0[int(x_shift1 * superSamp):, int(y_shift1 * superSamp):], superSamp,
        np.mean)
    yy_shift1 = rebin(
        yy0[int(x_shift1 * superSamp):, int(y_shift1 * superSamp):], superSamp,
        np.mean)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx_shift, yy_shift, g_shift, lw=0, cmap='viridis')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow()

    # try the magic kriging
    K = Kriging2d([1.169, 1.164], amp=108.74)
    gp = K.gp
    X = np.column_stack((np.concatenate((xx.flatten(), xx_shift1.flatten())),
                         np.concatenate((yy.flatten(), yy_shift1.flatten()))))
    Xpred = np.column_stack((xx_shift.flatten(), yy_shift.flatten()))
    gp = K.kriging(X, np.concatenate((g.flatten(), g_shift1.flatten())))
    K.kriging_mcmc(
        X, np.concatenate((g.flatten(), g_shift1.flatten())), plot=True)
    imGP = K.interpolate(Xpred, shape=xx_shift.shape)
    imBilinear = shift(g, [-x_shift, -y_shift], order=1)[:-1, :-1]
    imBicubic = shift(g, [-x_shift, -y_shift], order=3)[:-1, :-1]
    vmin = -.2
    vmax = .2
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(131)
    ax1.pcolormesh(
        xx_shift,
        yy_shift,
        imGP - g_shift,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax)
    ax1.set_title('GP Residual')
    ax2 = fig.add_subplot(132)
    ax2.pcolormesh(
        xx_shift,
        yy_shift,
        imBilinear - g_shift,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax)
    ax2.set_title('Bi-linear Interp Residual')
    ax3 = fig.add_subplot(133)
    ax3.pcolormesh(
        xx_shift,
        yy_shift,
        imBicubic - g_shift,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax)
    ax3.set_title('Bi-Cubic Interp Residual')
    for ax in [ax1, ax2, ax3]:
        ax.set_aspect('equal')
    print('Gaussian Process Residual: {0:.3f}'.format((imGP - g_shift).std()))
    print('Bilinear Residual: {0:.3f}'.format((imGP - imBilinear).std()))
    print('Bicubic Residual: {0:.3f}'.format((imGP - imBicubic).std()))
    plt.show()
    # K = kriging2d()
