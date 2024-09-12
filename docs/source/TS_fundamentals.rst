.. _ts_fundamentals:

Fundamentals of Thomson Scattering 
==========================================

This page provides some basics on Thomson scatering to help users figure out how to modify the input deck. 

**What is Thomson Scattering?** 
Thomson scattering is a diagnostic technique used to obtain information about the plasma conditions such as temperature and density.

**Lagmuir waves** are  high frequency electron plasma waves.

**Electron Plasma Waves (EPW)** also known as Langmuir waves, are high frequency electrostatic waves in a plasma with electrons as the oscillating species.

**Ion Acoustic Waves (IAW)** are low frequency electrostatic waves in a plasma, with ions as the primary oscillating species. 
In this case the electrons fight to sustain charge neutrality.

**Scattering angle** is the angle between the incident and scattered  electromagnetic wave.

Light scattered off of plasma waves experience **Doppler shifts** due to the velocity of electrons relative to the incoming light. 
These shifts provide insights into the movemet of electrons and characteristics of the plasma waves. 

Fit and data plot for EPW
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This plot gives an insight into the quality of the analysis, as it compares the fitted results (left) with the raw data (right).


.. image:: _elfolder/fit_and_data_ele.png
    :scale: 75%

Thomson scattering spectra for EPW 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The image provides a simplified visual representation of the effect of each parameter on the spectrum.  

.. image:: _elfolder/TS_spectra_EPW.JPG
    :scale: 75%

**Electron temperature (Te)**  as defined througth the averarge kinetic energy.

**Electron density (ne)** is the number of free electrons per unit volume,and it is used to characterize the plasma.

**m** is the super-Gaussian order of the electron velocity distribution function

**amp1** is the blue-shifted EPW amplitude multiplier.

**amp2** is the red-shifted EPW amplitude multiplier. 


Electron fit ranges plot 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **Electron Fit Ranges plot** uses lines to visually represent the region where data is being analyzed.

.. image:: _elfolder/electron_fit_annotated.png
    :scale: 75%

**Lineouts** are locations where data will be analyzed. 

**Lineout : start** specifies the first location where a lineout will be take

**lineout : end** specifies the last location where a lineout will be taken

Red abd blue,  min and max are the start and end of the region where the shift will be analyzed.

Min and max values for red and blue shift, are the start and end region whete the shifts will be analyzed. 

The scattered light is **red-shifted** if the electrons move away from the incoming light source. 
If the electrons move towards the light source, the scattered light is **blue-shifted**.  
**Min** and **max** values of these shifts indicate starting and ending waveleghts repectively to be analyzed.

Fit and data plot for IAW
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _elfolder/fit_and_data_iaw.png
    :scale: 75%
    

Thomson scattering spectra for IAW
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _elfolder/TS_spectra_IAW.JPG
    :scale: 75%

**Ion tempreature (Ti)** the ion temperature as defined through the average kinetic energy.

**Z** is the average ionization state. 

**Va** is the plasma flow velocity.

**lam** is the probe wavelenght. 

**ud** is the electron drfit velocity. 

**amp3** is the IAW amplitude multiplier.

Ion fit ranges plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This plot uses lines to visually represent the region where data is being analyzed.

.. image:: _elfolder/ion_fit_annotated.png
    :scale: 75%

**IAW max & min** designate the starting and ending wavelenghts for the analysis of the IAW.

**IAW cf min & max** indicate the starting and ending wavelenghts for a central feature to be excluded from analysis.




