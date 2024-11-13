from typing import Dict
import numpy as np
import scipy.io as sio
from os.path import join

from tsadar.data_handleing.calibrations.sa_table import sa_lookup


def get_calibrations(shotNum, tstype, t0, CCDsize):
    """
    Contains and loads the appropriate instrument calibrations based off the shot number and type of Thomson scattering
    performed. The calibrations loaded are the spectral dispersion, offset for the spectral axis, spectral instrument
    response functions (as the 1 standard deviation value), and a scale for the x-axis. In the case of temporal data
    this scale is the time per pixel. In the case of Imaging data the scale is a magnification and there is also an
    offset based off the TCC location. The calibrated axes are return as well as calibration values that will be needed
    later.

    For non-OMEGA data this function will have to be reworked.


    Args:
        shotNum: OMEGA shot number
        tstype: string with the ype of data, 'temporal', 'imaging', or 'angular'
        CCDsize: list with the CCD size in pixels, for OMEGA data this is [1024, 1024]

    Returns: return axisxE, axisxI, axisyE, axisyI, magE, stddev
        axisxE: Calibrated x-axis for electron data [time (ps), space(um), or scattering angle(degree)]
        axisxI: Calibrated x-axis for ion data [time (ps), space(um), or scattering angle(degree)]
        axisyE: Calibrated spectral/y-axis for electron data in nm
        axisyI: Calibrated spectral/y-axis for ion data in nm
        magE: scale for the x-axis (ps/px or um/px)
        stddev: dictionary with fields 'spect_stddev_ion' and 'spect_stddev_ele' containing the standard deviation
        (width) of the ion an electron spectral instrument response function respectively. In the case of angular data
        the fields 'spect_FWHM_ele' and 'ang_FWHM_ele' may be present containing the spectral and angular instrumental
        width in full-width-half-max.

    """
    stddev = dict()
    # Dispersions and calibrations
    if tstype == "angular":
        if shotNum < 95000:
            EPWDisp = 0.214116
            # EPWoff = 449.5272
            EPWoff = 449.5272
        elif shotNum < 105000:
            EPWDisp = 0.2129
            EPWoff = 439.8
        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.2129
            EPWoff = 439.8

        IAWDisp = 1  # dummy since ARTS does not measure ion spectra
        IAWoff = 1  # dummy
        stddev["spect_stddev_ion"] = 1  # dummy
        magE = 1  # dummy
        stddev["spect_FWHM_ele"] = 0.9  # nominally this is ~.8 or .9 for h2
        stddev["spect_stddev_ele"] = stddev["spect_FWHM_ele"] / 2.3548  # dummy
        stddev["ang_FWHM_ele"] = 1  # see Joe's FDR slides ~1-1.2
        # IAWtime = 0  # means nothing here just kept to allow one code to be used for both

    elif tstype == "temporal":
        if 98610 < shotNum < 98620:
            # These are valid for the 8-26-21 shot day, not sure how far back they are valid
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 522.894  # 522.90
            stddev["spect_stddev_ion"] = 0.0238  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        elif shotNum < 105000:
            # These are valid for the 8-26-21 shot day, not sure how far back they are valid
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 523.1  # 522.90
            stddev["spect_stddev_ion"] = 0.02262  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        elif shotNum < 108950:
            # these are calibrations for shot 108135
            EPWDisp = 0.4104
            IAWDisp = 0.005749
            EPWoff = 319.3
            IAWoff = 523.3438  # 522.90
            stddev["spect_stddev_ion"] = 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        elif shotNum < 108990:
            # these are calibrations for shots 108964-
            EPWDisp = 0.4104
            IAWDisp = 0.00959
            EPWoff = 135.0
            IAWoff = 346.09
            stddev["spect_stddev_ion"] = 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        elif 111410 < shotNum < 111435:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 317.4
            IAWoff = 522.92
            stddev["spect_stddev_ion"] = 0.0153  # 0.0095  # needs to be updated
            stddev["spect_stddev_ele"] = 0.668  # based of hg lamp data
            print("used 0.668 nm irf")
            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5.23  # (ps / px) this is just a rough guess
            magE = 5.35  # (ps / px) this is just a rough guess

        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 522.90
            stddev["spect_stddev_ion"] = 0.02262  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

    # IAWtime = 0  # temporal offset between EPW ross and IAW ross (varies shot to shot, can potentially add a fix based off the fiducials)

    else:
        if shotNum < 104000:
            EPWDisp = 0.27093
            IAWDisp = 0.00438
            EPWoff = 396.256  # needs to be checked
            IAWoff = 524.275

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.4365  # needs to be checked

            magI = 2.87  # um / px
            magE = 5.10  # um / px

            EPWtcc = 1024 - 456.1  # 562;
            IAWtcc = 1024 - 519  # 469;

        elif 106303 <= shotNum <= 106321:
            # refractive teloscope used on 11/8/22
            EPWDisp = 0.27594
            IAWDisp = 0.00437
            EPWoff = 388.256  # 390.256 worked for 106317
            IAWoff = 524.345

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.4365  # needs to be checked

            magI = 2.89 / 0.3746 * 1.118  # um / px times strech factor accounting for tilt in view
            magE = 5.13 / 0.36175 * 1.118  # um / px times strech factor accounting for tilt in view

            EPWtcc = 1024 - 503  # 562;
            IAWtcc = 1024 - 450 #578  # 469;

        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.27093
            IAWDisp = 0.00437
            EPWoff = 396.256  # needs to be checked
            IAWoff = 524.275

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.4365  # needs to be checked

            magI = 2.89 * 1.079  # um / px times strech factor accounting for tilt in view
            magE = 5.13 * 1.079  # um / px times strech factor accounting for tilt in view

            EPWtcc = 1024 - 516  # 562;
            IAWtcc = 1024 - 450  # 469;

        # IAWtime = 0  # means nothing here just kept to allow one code to be used for both

    ## Apply calibrations
    axisy = np.arange(1, CCDsize[0] + 1)
    axisyE = axisy * EPWDisp + EPWoff  # (nm)
    axisyI = axisy * IAWDisp + IAWoff  # (nm)

    if tstype != "angular":
        axisx = np.arange(1, CCDsize[1] + 1)
        axisxE = (axisx - t0[1]) * magE  # ps,um
        axisxI = (axisx - t0[0]) * magI  # ps,um
        if tstype == "imaging":
            axisxE = axisxE - EPWtcc * magE
            axisxI = axisxI - IAWtcc * magI
            # axisxI = axisxI + 200
    else:
        imp = sio.loadmat(join("files", "angsFRED.mat"), variable_names="angsFRED")
        axisxE = imp["angsFRED"][0, :]
        # axisxE = np.vstack(np.loadtxt("files/angsFRED.txt"))
        axisxI = np.arange(1, CCDsize[1] + 1)

    return axisxE, axisxI, axisyE, axisyI, magE, stddev


def get_scattering_angles(config: Dict) -> Dict:
    """
    Loads and returns a scattering angle dictionary based off the input deck. The scattering angle dictionary has 2
    fields 'sa' and 'weights'. The field 'sa' is an array of the scattering angles present based off the geometry
    specified in the input deck. Multiple scattering angles are present due to the finite size of the apertures. The
    field 'weights' is an array of the same size as 'sa' with the relative weights of each scattering angle in the final
    spectrum.

    Known geometries are for OMEGA and more would need to be added for another system.


    Args:
        config: Dictionary built from the input deck

    Returns:
        sa: Dictionary with scattering angles and weights

    """
    if config["other"]["extraoptions"]["spectype"] != "angular":
        sa = sa_lookup(config["data"]["probe_beam"])
    else:
        # Scattering angle in degrees for Artemis
        imp = sio.loadmat(join("files", "angleWghtsFredfine.mat"), variable_names="weightMatrix")
        weights = imp["weightMatrix"]
        sa = dict(sa=np.arange(19, 139.5, 0.5), weights=weights)
    return sa
