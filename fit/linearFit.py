import numpy as np
import matplotlib.pyplot as plt


def linearFit(x, y, dy=None, doPlot=False, ax=None):
    if dy is None:
        dy = np.ones(len(y))
    """
    my own linear fit routine, since there is no good scipy or numpy linearFit routine written up
    """
    Y = np.mat(y).T
    A = np.mat([np.ones(len(x)), x]).T
    C = np.mat(np.diagflat(dy**2))
    mat1 = (A.T * C**(-1) * A)**(-1)
    mat2 = A.T * C**(-1) * Y
    b, m = mat1 * mat2
    b = b.flat[0]
    m = m.flat[0]
    sigb, sigm = np.sqrt(np.diag(mat1))
    chisq = 1. / (len(x) - 2) * np.sum((y - m * x - b)**2 / dy**2)
    if doPlot:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.plot(x, y, '.')
        ax.plot((min(x), max(x)), (m*min(x) + b, m*max(x) + b))
        ax.text(0.6, 0.1, r'$y={0:.2f}x + {1:.2f}$'.format(m, b),
                transform=ax.transAxes)
    return b, m, sigb, sigm, chisq
