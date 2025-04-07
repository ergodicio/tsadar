import numpy as np
import matplotlib.pyplot as plt
import math, os
from os.path import join, exists


BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "external")



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

    warp1r = np.sqrt(warp1x**2 + warp1y**2)

    print("dewarping epw")
    depimg = np.zeros(np.shape(warpedData))
    lenarrray = np.zeros(np.shape(warpedData))
    lenarrrayx = np.zeros(np.shape(warpedData))
    lenarrrayy = np.zeros(np.shape(warpedData))

    for i in range(len(warpedData)):
        for j in range(len(warpedData[0])):
            rawpoint = np.array([j, i])
            # transpoint=np.array([j+warp1y[i,j],i+warp1x[i,j]])
            transpoint = np.array([j + warp1y[j, i], i + warp1x[j, i]])

            txpix = transpoint[1]
            typix = transpoint[0]
            valold = warpedData[j, i]

            diff = np.sqrt(np.sum((transpoint - rawpoint) ** 2))
            diffy = transpoint[0] - rawpoint[0]
            diffx = transpoint[1] - rawpoint[1]
            xl = math.floor(txpix)
            xh = math.ceil(txpix)
            yl = math.floor(typix)
            yh = math.ceil(typix)
            xlf = 1.0 - (txpix - xl)
            ylf = 1.0 - (typix - yl)
            if yl > 0 and xl > 0:
                try:
                    depimg[yl, xl] = depimg[yl, xl] + valold * xlf * ylf
                    depimg[yl, xh] = depimg[yl, xh] + valold * (1 - xlf) * ylf
                    depimg[yh, xl] = depimg[yh, xl] + valold * xlf * (1 - ylf)
                    depimg[yh, xh] = depimg[yh, xh] + valold * (1 - xlf) * (1 - ylf)
                except:
                    offstr = "off"
            lenarrray[j, i] = diff
            lenarrrayx[j, i] = diffx
            lenarrrayy[j, i] = diffy

    # %%%%%%%%%%%%%%%%%
    # fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    # imI = ax[0].imshow(warpedData, vmax=100)
    # imI = ax[1].imshow(depimg, vmax=100)
    # imI = ax[2].imshow(warpedData-depimg, vmin=-100,vmax=100)
    # plt.show()
    dewarped = depimg
    print("epw dewarped")

    return dewarped
