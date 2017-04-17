import numpy as np

def circleMask(shape, maskList, unMaskList=None):
    """
mask list of rings, each ring represent by centroid,
inner radius and outer radius
"""
    mask = np.zeros(shape, dtype=int)
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[1]))
    for circle in maskList:
        dist = (xx-circle[0])**2 + (yy-circle[1])**2
        mask[np.where((dist >= circle[2]**2) & (dist <= circle[3]**2))] = 1
    if unMaskList is not None:
        for circle in unMaskList:
            dist = (xx-circle[0])**2 + (yy-circle[1])**2
            mask[np.where((dist >= circle[2]**2) & (dist <= circle[3]**2))] = 0
    return mask.astype(bool)
