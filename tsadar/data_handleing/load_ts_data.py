from os.path import join
from os import listdir
import os
from pyhdf.SD import SD, SDC
import numpy as np
from scipy.signal import find_peaks
from tsadar.process.warpcorr import perform_warp_correction

BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "aux")


def loadData(sNum, sDay, loadspecs, custom_path=False):
    """
        This function loads the appropriate data based off the provided shot number (sNum) automatically determining the
        type of data in the file. The flag sDay changes the default path to the temporary archive on the redwood server and
        will only work if connected to the LLE redwood server (depreciated).

        If the data is time resolved an attempt will be made to locate t=0 based off the fiducials. The relationship
        between the fiducials and t=0 should be fixed, with the data potentialy moving if delay is added to the probe.

        Known Issues:
            If there are mutliple data types from the same shot number such as ATS and imaging data, this algorithm will
            fail.
            This only works for OMEGA data and will need to be reworked for non-OMEGA data.

        Args:
            sNum: Shot number
            sDay: N/A
            loadspecs: Dictionary containing the options of which spectra should be loaded, sub-dictionary of the input
                deck
    eDat, iDat, xlab, t0, specType
        Returns:
            eDat: electron data
            iDat: ion data
            xlab: x-axis label, one of the following: Radius, Time, Angle
            t0: pixel location of t0 in the ion and electron data, defaults to 0 if no fiducials are found

    """
    if sDay:
        folder = r"\\redwood\archive\tmp\thomson"
    elif custom_path:
        folder = custom_path
    else:
        folder = join(BASE_FILES_PATH, "data")

    print(f"{folder=}")
    file_list = listdir(folder)
    files = [name for name in file_list if str(sNum) in name]
    t0 = [0, 0]

    for fl in files:
        if "epw" in fl or "EPW" in fl:
            hdfnameE = join(folder, fl)
            if "ccd" in fl or "CCD" in fl:
                xlab = "Radius (\mum)"
                specType = "imaging"
            else:
                xlab = "Time (ps)"
                specType = "temporal"
        if "iaw" in fl or "IAW" in fl:
            hdfnameI = join(folder, fl)
            if "ccd" in fl or "CCD" in fl:
                xlab = "Radius (\mum)"
                specType = "imaging"
            else:
                xlab = "Time (ps)"
                specType = "temporal"
        if "ats" in fl or "ATS" in fl:
            hdfnameE = join(folder, fl)
            specType = "angular"
            xlab = "Scattering angle (degrees)"

    if loadspecs["load_ion_spec"]:
        try:
            iDatfile = SD(hdfnameI, SDC.READ)
            sds_obj = iDatfile.select("Streak_array")  # select sds
            iDat = sds_obj.get()  # get sds data
            iDat = iDat.astype("float64")
            iDat = iDat[0, :, :] - iDat[1, :, :]
            iDat = np.flipud(iDat)

            if specType == "imaging":
                iDat = np.rot90(np.squeeze(iDat), 3)
            elif loadspecs["absolute_timing"]:
                # this sets t0 by locating the fiducial and placing t0 164px earlier
                fidu = np.sum(iDat[850:950, :], 0)
                res = find_peaks(fidu, prominence=1000, width=10)
                peak_center = res[1]["left_ips"][0] + (res[1]["right_ips"][0] - res[1]["left_ips"][0]) / 2.0
                t0[0] = round(peak_center - 164)
        except BaseException:
            print("Unable to find IAW")
            iDat = []
            loadspecs["load_ion_spec"] = False
    else:
        iDat = []

    print(f"{hdfnameE=}")

    if loadspecs["load_ele_spec"]:
        try:
            eDatfile = SD(hdfnameE, SDC.READ)
            sds_obj = eDatfile.select("Streak_array")  # select sds
            eDat = sds_obj.get()  # get sds data
            eDat = eDat.astype("float64")
            eDat = eDat[0, :, :] - eDat[1, :, :]

            if specType == "angular":
                eDat = np.fliplr(eDat)
                print("found angular data")
            elif specType == "temporal":
                eDat = perform_warp_correction(eDat)
            elif specType == "imaging":
                eDat = np.rot90(np.squeeze(eDat), 3)

            if specType == "temporal" and loadspecs["absolute_timing"]:
                # this sets t0 by locating the fiducial and placing t0 164px earlier
                fidu = np.sum(eDat[0:100, :], 0)
                res = find_peaks(fidu, prominence=1000, width=10)
                peak_center = res[1]["left_ips"][0] + (res[1]["right_ips"][0] - res[1]["left_ips"][0]) / 2.0
                t0[1] = round(peak_center - 95)
        except BaseException:
            print("Unable to find EPW")
            eDat = []
            loadspecs["load_ele_spec"] = False
    else:
        eDat = []

    if not loadspecs["load_ele_spec"] and not loadspecs["load_ion_spec"]:
        raise LookupError(f"No data found for shotnumber {sNum} in the data folder")

    return eDat, iDat, xlab, t0, specType
