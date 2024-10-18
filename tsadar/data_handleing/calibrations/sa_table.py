import numpy as np


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
        raise NotImplementedError("Other probe geometrries are not yet supported")

    return sa
