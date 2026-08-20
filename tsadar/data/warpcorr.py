"""Dewarps EPW streak-camera images using a precomputed pixel-displacement map, correcting for the sweep
optics' geometric distortion."""
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, exists

BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "external")


def perform_warp_correction(warpedData, instrument="EPW", sweepSpeed=5, flatField=True):
    """
    Returns a dewarped streak camera image. Currently this only works for %ns EPWs and does not have flatfileds but other versions will be added (10/28/22).

    Args:
        warpedData: The streak camera image to be dewarped
        instrument: 'EPW' or 'IAW' corresponding to the diangostic instrument
        sweepSpeed: sweep time in ns based on camera settings
        flatField: Flag to use flatfiled data for a flat field correction

    Returns:
        dewarped: The dewarped data

    """

    if instrument == "EPW":
        if sweepSpeed == 5:
            warp1x = np.load(join(BASE_FILES_PATH, "files", "epwtestDW5img1x.npy"))
            warp1y = np.load(join(BASE_FILES_PATH, "files", "epwtestDW5img1y.npy"))
        # elif sweepSpeed == 15:

        else:
            warp1x = np.load(join(BASE_FILES_PATH, "files", "epwtestDW5img1x.npy"))
            warp1y = np.load(join(BASE_FILES_PATH, "files", "epwtestDW5img1y.npy"))
            print("no specific data avaiable for this sweep speed - using 5ns dewarp")
    else:
        raise NotImplementedError(f"perform_warp_correction currently only supports instrument='EPW', got {instrument!r}")

    warp1r = np.sqrt(warp1x**2 + warp1y**2)

    print("dewarping epw")
    ny, nx = warpedData.shape
    depimg = np.zeros((ny, nx))

    # I, J mirror the original nested "for i ... for j ..." loop: I is the source column
    # index (0..ny-1, used as the "i" of warpedData[j, i]), J is the source row index
    # (0..nx-1, used as its "j"). Kept as separate names to match the original [j, i]
    # indexing convention rather than the more natural [row, col].
    I, J = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    valold = warpedData[J, I]
    txpix = I + warp1x[J, I]
    typix = J + warp1y[J, I]

    xl = np.floor(txpix).astype(int)
    xh = np.ceil(txpix).astype(int)
    yl = np.floor(typix).astype(int)
    yh = np.ceil(typix).astype(int)
    xlf = 1.0 - (txpix - xl)
    ylf = 1.0 - (typix - yl)

    base_mask = (yl > 0) & (xl > 0)

    def _scatter(yidx, xidx, weight):
        # Each of the 4 bilinear-splat terms is bounds-checked independently, unlike the
        # original's single try/except around all 4 writes (where one out-of-bounds term
        # silently dropped the remaining terms for that pixel too).
        valid = base_mask & (yidx >= 0) & (yidx < ny) & (xidx >= 0) & (xidx < nx)
        np.add.at(depimg, (yidx[valid], xidx[valid]), (valold * weight)[valid])

    _scatter(yl, xl, xlf * ylf)
    _scatter(yl, xh, (1 - xlf) * ylf)
    _scatter(yh, xl, xlf * (1 - ylf))
    _scatter(yh, xh, (1 - xlf) * (1 - ylf))

    # %%%%%%%%%%%%%%%%%
    # fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    # imI = ax[0].imshow(warpedData, vmax=100)
    # imI = ax[1].imshow(depimg, vmax=100)
    # imI = ax[2].imshow(warpedData-depimg, vmin=-100,vmax=100)
    # plt.show()
    dewarped = depimg
    print("epw dewarped")

    return dewarped
