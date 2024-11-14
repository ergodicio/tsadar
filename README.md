# TSADAR
`TSADAR` performs Thomson Scattering analysis using Automatic Differentiation (AD) and GPUs (if available). At this time, it is heavily specialized towards analyzing data 
from OMEGA experiments at the Laboratory for Laser Energetics. However, there is no reason this cannot be extended to work with data
from other facilities

## Thomson Scattering
-- work in progress -- 

## Installation
If cloning the respository onto a windows machine it may be necessary to modify the git config `git config --global core.protectNTFS false`. This is multistep for now, at least on Mac, because `pyhdf` using `pip` has some problems. We can get around that by using conda

 - Install conda 
 - Make conda environment for `tsadar`
 - Install `tsadar` using `pip install https<>`
 - Install `pyhdf` using `conda`

## Documentation
Go to https://tsadar.readthedocs.io/ for detailed documentation.

## Automatic Differentiation
In Thomson Scattering, as in other parameter estimation inverse problems, there can be many parameters. In the case where the forward model is known, 
gradient-based methods can be applied to solve this many parameter optimization problem. Automatic Differentiation (AD) enables fast and efficient calculation of (relatively) arbitrary numerical programs. Here, we apply it to the form factor calculation.

## Citation
1. Milder, A. L., Joglekar, A. S., Rozmus, W. & Froula, D. H. Qualitative and quantitative enhancement of parameter estimation for model-based diagnostics using automatic differentiation with an application to inertial fusion. Mach. Learn.: Sci. Technol. 5, 015026 (2024).


