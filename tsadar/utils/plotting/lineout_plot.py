import matplotlib.pyplot as plt
import numpy as np
import os


def lineout_plot(data, fits, sqdev, yaxis, ylim, s_ind, e_ind, titlestr, filename, td, tag):
    """
    Plots lineouts of data and their fits, along with residuals, and saves the figure to a file.
    Parameters:
        data (list of np.ndarray): List containing data arrays to plot. The list can have one or 2 elements being the electron and ion data.
        fits (list of np.ndarray): List containing fit arrays corresponding to the data. Must be the same shape as data.
        sqdev (list of np.ndarray): List containing squared deviation (residual) arrays. Must be the same shape as data.
        yaxis (list of np.ndarray): List containing y-axis (e.g., wavelength) arrays for each plot.
        ylim (tuple): Tuple specifying the y-axis limits for the data plots.
        s_ind (list of int): List of start indices to start the plotting, based of the wavelength set in the default deck.
        e_ind (list of int): List of end to end the plotting, based of the wavelength set in the default deck
        titlestr (str): Title string for the plots.
        filename (str): Name of the file to save the figure as.
        td (str): Directory path where the figure will be saved.
        tag (str): String denoting which lineouts are being plotted, the "best" or "worst"
    Returns:
        None: The function saves the plots to a file in the specified directory.
    Notes:
        - The function creates a 2-row subplot where the first row contains the data and fit plots, and the second row contains the residuals.
        - The function uses matplotlib for plotting and saves the figure to a specified directory.
        - The function handles both electron and ion data if provided.
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
