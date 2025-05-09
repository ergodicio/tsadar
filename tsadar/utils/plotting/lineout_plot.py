import matplotlib.pyplot as plt
import numpy as np
import os


def lineout_plot(data, fits, sqdev, yaxis, ylim, s_ind, e_ind, titlestr, filename, td, tag):
    """
    Plots lineout comparing the fits to the data. If the data has both electron and ion data they will both be plotted.
    The value of the fit metric chi^2 per point is plotted beneath the data and fit.


    Args:
        data: Processed data (spectrum) to be plotted as a list of arrays. If the list can have one or 2 elements being
            the electron and ion data.
        fits: Fits results (spectrum) to be plotted as a list of arrays. Must be the same shape as data.
        sqdev: chi^2 per point. Must be the same shape as data
        yaxis: Spectral axis, same shape as data
        s_ind: Index to start the plotting, based of the wavelength set in the default deck
        e_ind: Index to end the plotting, based of the wavelength set in the default deck
        titlestr: string to be used as the title of the plot
        filename: string to be used as the name of the file where the plots will be saved
        td: temporary directory that will be uploaded to mlflow
        tag: string denoting which lineouts are being plotted, the "best" or "worst"

    Returns:

    """
    if len(data) == 2:
        num_col = 2
    else:
        num_col = 1

    fig, ax = plt.subplots(2, num_col, figsize=(12, 8), squeeze=False, tight_layout=True, sharex=False)
    for col in range(num_col):
        ax[0][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]], np.squeeze(data[col][s_ind[col] : e_ind[col]]), label="Data"
        )
        ax[0][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]], np.squeeze(fits[col][s_ind[col] : e_ind[col]]), label="Fit"
        )

        ax[0][col].set_title(titlestr, fontsize=14)
        ax[0][col].set_ylabel("Amp (arb. units)")
        ax[0][col].legend(fontsize=14)
        ax[0][col].grid()
        ax[0][col].set_ylim(ylim)

        ax[1][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]], np.squeeze(sqdev[col][s_ind[col] : e_ind[col]]), label="Residual"
        )
        ax[1][col].set_xlabel("Wavelength (nm)")
        ax[1][col].set_ylabel(r"$\chi_i^2$")

    fig.savefig(os.path.join(td, tag, filename), bbox_inches="tight")
    plt.close(fig)
