import os
import numpy as np
from flatten_dict import flatten, unflatten
import yaml

from ..core.thomson_diagnostic import ThomsonScatteringDiagnostic
from ..core.modules.ts_params import ThomsonParams
from .fitter import _validate_inputs_, load_data_for_fitting
from ..data.calibration import get_calibrations
from .loss_function import LossFunction

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_measured_data(fig, all_data, all_axes, config):
    if config["data"]["load_ele_spec"]:
        fig.add_trace(
            go.Scatter(
                x=all_axes["epw_y"].squeeze(),
                y=all_data["e_data"].squeeze(),
                mode="lines",
                name="Measured EPW",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Amp (arb. units)", row=1, col=1)
    if config["data"]["load_ion_spec"]:
        fig.add_trace(
            go.Scatter(
                x=all_axes["iaw_y"].squeeze(),
                y=all_data["i_data"].squeeze(),
                mode="lines",
                name="Measured IAW",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
        fig.update_yaxes(title_text="Amp (arb. units)", row=1, col=2)

        fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
        fig.update_yaxes(title_text="Chi sq", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Minimization Loss", row=2, col=1, secondary_y=True)

        fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=2)
        fig.update_yaxes(title_text="Chi sq", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Minimization Loss", row=2, col=2, secondary_y=True)

def gen_and_plot_theory(ts_diag, loss_fn, batch, config, fig):
    
    ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
    ThryE, ThryI, lamAxisE, lamAxisI = ts_diag(ts_params, batch)

    if config["data"]["load_ele_spec"]:
        fig.add_trace(
            go.Scatter(
                x=lamAxisE.squeeze(),
                y=ThryE.squeeze(),
                mode="lines",
                name="Simulated EPW",
            ),
            row=1,
            col=1,
        )
    if config["data"]["load_ion_spec"]:
        fig.add_trace(
            go.Scatter(
                x=lamAxisI.squeeze(),
                y=ThryI.squeeze(),
                mode="lines",
                name="Simulated IAW",
            ),
            row=1,
            col=2,
        )
    
    denom = [ThryE+37.0*(2*config["data"]["dpixel"]+1), ThryI+35.0*(2*config["data"]["dpixel"]+1)]
    i_error, e_error, sqdev = loss_fn.calc_ei_error(
        batch,
        ThryI,
        lamAxisI,
        ThryE,
        lamAxisE,
        denom,
        reduce_func = np.nanmean,
    )
    # i_tot_error = np.sum(i_error)
    # e_tot_error = np.sum(e_error)

    fig.add_trace(
        go.Scatter(
            x=lamAxisE.squeeze(),
            y=sqdev["ele"].squeeze(),
            mode="lines",
            name="Real Chisq",
            text=f"Total EPW Chisq: {e_error}",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=lamAxisI.squeeze(),
            y=sqdev["ion"].squeeze(),
            mode="lines",
            name="Real Chisq",
            text=f"Total IAW Chisq: {i_error}",
        ),
        row=2,
        col=2,
        secondary_y=False,
    )

    print(f"Total EPW Chi sq: {e_error}, Total IAW Chi sq: {i_error}")

    denom=[np.abs(np.amax(batch["i_amps"])), np.abs(np.amax(batch["e_amps"]))]
    i_error, e_error, sqdev = loss_fn.calc_ei_error(
        batch,
        ThryI,
        lamAxisI,
        ThryE,
        lamAxisE,
        denom,
        reduce_func = np.nanmean,
    )

    fig.add_trace(
        go.Scatter(
            x=lamAxisE.squeeze(),
            y=sqdev["ele"].squeeze(),
            mode="lines",
            name="Minimization Loss",
            text=f"Total EPW Minimization Loss: {e_error}",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=lamAxisI.squeeze(),
            y=sqdev["ion"].squeeze(),
            mode="lines",
            name="Minimization Loss",
            text=f"Total IAW Minimization Loss: {i_error}",
        ),
        row=2,
        col=2,
        secondary_y=True,
    )
    fig.update_layout(yaxis4=dict(overlaying='y3', side='right', tickmode='sync'))
    fig.update_layout(yaxis6=dict(overlaying='y5', side='right', tickmode='sync'))

    print(f"Total EPW Minimization Loss: {e_error}, Total IAW Minimization Loss: {i_error}")
    return ThryE, ThryI, lamAxisE, lamAxisI

def forward_pass(config):
    
    """
    Calculates a spectrum and compares it to data with the ability to try multiple spectra and visualize the spectral components. Includes the ability to turn on and off various instrament effects.
    This only works with 1D spectra and a single lineout at a time. The single lineout is the first lineout as specified using the normal lineout specification in the config file.
    This is only intended to run localy for interactive use and not for batch runs.
    """

    #hard coded options for interactive forward pass
    ksmear= False  #if True includes k-smearing in the forward model
    include_IRF=True
    background_subtract=False #if True the BG is subtracted from the data before plotting, if false BG is added to the fit model
    update_fig = False  #if true the figure is updated after each forward pass with curves being replaced, if false a new curve is added each time
    
    config = _validate_inputs_(config)

    # prepare data
    all_data, sas, all_axes = load_data_for_fitting(config)
    
    is_angular = True if "angular" in config["other"]["extraoptions"]["spectype"] else False
    config["other"]["extraoptions"]["spectype"] += "_interactive"
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1

    if not is_angular:
        if not ksmear:
            sas['sa'] = np.array([154.0])  # hardcoded for now
            sas['weights'] = np.array([1.0])
        else:
            sas['sa'] = sas['sa']
            sas['weights'] = sas['weights']
    batch = {
                "e_data": all_data["e_data"][0]-all_data["noiseE"][0] if background_subtract else all_data["e_data"][0],
                "e_amps": all_data["e_amps"][0],
                "i_data": all_data["i_data"][0]-all_data["noiseI"][0] if background_subtract else all_data["i_data"][0],
                "i_amps": all_data["i_amps"][0],
                "noise_e": all_data["noiseE"][0] if not background_subtract else 0.0,
                "noise_i": all_data["noiseI"][0] if not background_subtract else 0.0,
            }

    fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": False}, {"secondary_y": False}], [{"secondary_y": True}, {"secondary_y": True}]],
                        subplot_titles=("Electron Spectrum", "Ion Spectrum", "Electron Chisq",  "Ion Chisq"))
    plot_measured_data(fig, batch, all_axes, config)

    ts_diag = ThomsonScatteringDiagnostic(config, scattering_angles=sas)
    #ts_params = ThomsonParams(config["parameters"], num_params=1, batch=False)
    loss_fn = LossFunction(config, sas, batch)
    ThryE, ThryI, lamAxisE, lamAxisI = gen_and_plot_theory(ts_diag, loss_fn, batch, config, fig)

    fig.show()
    cont = 'y'
    while cont.casefold() not in ["n", "no"]:
        cont = input("Do you wish to continue? (y/n): ")
        if cont.casefold() in ["y", "yes"]:
            print("reloading config and continueing forward pass...")
            if update_fig:
                fig.data = [fig.data[0],fig.data[1]]  #clear the figure

            all_configs = {}
            basedir = os.path.join(os.getcwd(), "configs/1d")
            for k in ["defaults", "inputs"]:
                with open(f"{os.path.join(basedir, k)}.yaml", "r") as fi:
                    all_configs[k] = yaml.safe_load(fi)
            defaults = flatten(all_configs["defaults"])
            defaults.update(flatten(all_configs["inputs"]))
            config = unflatten(defaults)
            ThryE, ThryI, lamAxisE, lamAxisI = gen_and_plot_theory(ts_diag, loss_fn, batch, config, fig)
            fig.show()
        else:
            print("Exiting interactive forward pass.")
            return 
            
         
