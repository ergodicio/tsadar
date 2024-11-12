import xarray as xr
from matplotlib import figure, pyplot as plt

with xr.open_dataset("ele_fit_and_data (1).nc") as plot_example:
 ele_plot=plot_example,

fig, ax =plt.subplots(1,1,ele_plot)
figure = plt()

