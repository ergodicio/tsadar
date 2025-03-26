from typing import Dict
import numpy as np
import scipy.io as sio
import os

BASE_FILES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "external")


def sa_lookup(beam):
    """
    Creates the scattering angle dictionary with the scattering angles and their weights based of the chosen probe
    beam. All values are precalculated. Available options are P9, B12, B15, B23, B26, B35, B42, B46, B58.

    Args:
        beam: string with the name of the beam to be used as a probe

    Returns:
        sa: dictionary with scattering angles in the 'sa' field and their relative weights in the 'weights' field
    """
    if beam == "P9":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(53.637560, 66.1191, 10),
            weights=np.array(
                [
                    0.00702671050853565,
                    0.0391423809738300,
                    0.0917976667717670,
                    0.150308544660150,
                    0.189541011666141,
                    0.195351560740507,
                    0.164271879645061,
                    0.106526733030044,
                    0.0474753389486960,
                    0.00855817305526778,
                ]
            ),
        )
    elif beam == "B12":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(71.0195, 83.3160, 10),
            weights=np.array(
                [
                    0.007702,
                    0.0404,
                    0.09193,
                    0.1479,
                    0.1860,
                    0.1918,
                    0.1652,
                    0.1083,
                    0.05063,
                    0.01004,
                ]
            ),
        )
    elif beam == "B15":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(12.0404, 24.0132, 10),
            weights=np.array(
                [
                    0.0093239,
                    0.04189,
                    0.0912121,
                    0.145579,
                    0.182019,
                    0.188055,
                    0.163506,
                    0.1104,
                    0.0546822,
                    0.0133327,
                ]
            ),
        )
    elif beam == "B23":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(72.281, 84.3307, 10),
            weights=np.array(
                [
                    0.00945903,
                    0.0430611,
                    0.0925634,
                    0.146705,
                    0.182694,
                    0.1881,
                    0.162876,
                    0.109319,
                    0.0530607,
                    0.0121616,
                ]
            ),
        )
    elif beam == "B26":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(55.5636, 68.1058, 10),
            weights=np.array(
                [
                    0.00648619,
                    0.0386019,
                    0.0913923,
                    0.150489,
                    0.190622,
                    0.195171,
                    0.166389,
                    0.105671,
                    0.0470249,
                    0.00815279,
                ]
            ),
        )
    elif beam == "B35":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(32.3804, 44.6341, 10),
            weights=np.array(
                [
                    0.00851313,
                    0.0417549,
                    0.0926084,
                    0.149182,
                    0.187019,
                    0.191523,
                    0.16265,
                    0.106842,
                    0.049187,
                    0.0107202,
                ]
            ),
        )
    elif beam == "B42":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(155.667, 167.744, 10),
            weights=np.array(
                [
                    0.00490969,
                    0.0257646,
                    0.0601324,
                    0.106076,
                    0.155308,
                    0.187604,
                    0.19328,
                    0.15702,
                    0.0886447,
                    0.0212603,
                ]
            ),
        )
    elif beam == "B46":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(56.5615, 69.1863, 10),
            weights=np.array(
                [
                    0.00608081,
                    0.0374307,
                    0.0906716,
                    0.140714,
                    0.191253,
                    0.197333,
                    0.166164,
                    0.106121,
                    0.0464844,
                    0.0077474,
                ]
            ),
        )
    elif beam == "B58":
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(119.093, 131.666, 10),
            weights=np.array(
                [
                    0.00549525,
                    0.0337372,
                    0.0819783,
                    0.140084,
                    0.186388,
                    0.19855,
                    0.174136,
                    0.117517,
                    0.0527003,
                    0.00941399,
                ]
            ),
        )
    else:
        raise NotImplmentedError("Other probe geometrries are not yet supported")

    return sa


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
            stddev["spect_stddev_ele"] = 1.1024  # needs to be checked

            magI = 2.89 / 0.3746 * 1.118  # um / px times strech factor accounting for tilt in view
            magE = 5.13 / 0.36175 * 1.118  # um / px times strech factor accounting for tilt in view

            EPWtcc = 1024 - 503  # 562;
            IAWtcc = 1024 - 568  # 578  # 469;

        elif 107620 <= shotNum <= 107633:
            # refractive teloscope used on 3/9/23
            EPWDisp = 0.27594
            IAWDisp = 0.005701
            EPWoff = 388.256  # 390.256 worked for 106317
            IAWoff = 524.345

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.1024  # needs to be checked

            magI = 2.89 / 0.3746 * 1.118  # um / px times strech factor accounting for tilt in view
            magE = 5.13 / 0.36175 * 1.118  # um / px times strech factor accounting for tilt in view

            EPWtcc = 1024 - 503  # 562;
            IAWtcc = 1024 - 568  # 578  # 469;

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
        imp = sio.loadmat(os.path.join(BASE_FILES_PATH, "files", "angsFRED.mat"), variable_names="angsFRED")
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
        imp = sio.loadmat(
            os.path.join(BASE_FILES_PATH, "files", "angleWghtsFredfine.mat"), variable_names="weightMatrix"
        )
        weights = imp["weightMatrix"]
        sa = dict(sa=np.arange(19, 139.5, 0.5), weights=weights)
    return sa
