import sep
from astropy.io import fits
import numpy as np
from plot import imshow
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class sourceExtraction:
    def __init__(self, im, sigma=2.0, subback=False):
        self.im = im
        bkg = sep.Background(im)
        if subback:
            self.im = self.im - bkg.back()
        self.obj = sep.extract(self.im, sigma, err=bkg.globalrms)
        self.mask = None

    def mask_source(self, scale=1):
        if self.mask is not None:
            return self.mask
        yy, xx = np.mgrid[0:self.im.shape[0], 0:self.im.shape[1]]
        flux_scale = np.log(self.obj['flux']) / 2
        # cxx, cyy, cxy = sep.ellipse_coeffs(self.obj['a'], self.obj['b'], self.obj['theta'])
        self.mask = np.zeros_like(self.im, dtype=bool)
        for i in range(len(self.obj)):
            self.mask[np.where(self.obj['cxx'][i] * (xx - self.obj['x'][i])**2 +
                               self.obj['cyy'][i] * (yy - self.obj['y'][i])**2 +
                               self.obj['cxy'][i] * (xx - self.obj['x'][i]) * (yy - self.obj['y'][i])
                               < (scale*flux_scale[i])**2)] = True
        return self.mask

    def mark_source(self, scale=1, vmin=None, vmax=None):
        fig, ax = plt.subplots()
        imshow(self.im, ax=ax, vmin=vmin, vmax=vmax)
        flux_scale = np.log(self.obj['flux']) / 2
        for i in range(len(self.obj)):
            e = Ellipse(xy=(self.obj['x'][i], self.obj['y'][i]),
                width=scale * flux_scale[i] * self.obj['a'][i],
                height=scale * flux_scale[i] * self.obj['b'][i],
                angle=self.obj['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('C1')
            ax.add_artist(e)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    im = fits.getdata('icytb0xhq_flt.fits', 'sci').byteswap().newbyteorder()
    err = fits.getdata('icytb0xhq_flt.fits', 'err').byteswap().newbyteorder()
    ims = sourceExtraction(im, err, sigma=3.0, subback=True)
    source_mask = ims.mask_source(1)
    plt.close('all')
    # imshow(np.ma.array(im, mask=source_mask), vmax=100)
    ims.mark_source(2.5, vmax=100)
    plt.show()
