FAQ
---------------------------------

**Have more questions?**

Please file an issue at https://github.com/ergodicio/tsadar/issues and we can chat there!

.. dropdown:: What does TSADAR stand for?
   :color: primary 

   TSADAR stands for Thomson Scattering with Automatic Differentiation for Analysis and Regression


.. dropdown:: What is Thomson Scattering? 
   :color: primary 
    
   Thomson scattering is a diagnostic technique used to obtain information about the plasma conditions such as temperature and density.


.. dropdown:: How is TSADAR pronounced?
   :color: primary 
    
   It is pronounced like radar with a TS sound.


.. dropdown:: I'm having trouble fitting my data, what do I do?
   :color: primary 
    
   I wish there was a simple answer to this problem, but it depends on the specifics of your data. But here are the first things to double check. 
      
   First, does your fitting range match your data? If sections of the data are not visible to the minimizer (outside the fit ranges) the fit will likely fail, and if too much background is visible the minimizer may become preocupied with fitting the noise. Also the code tends to struggle when asked to fit lineouts with no data and then transition to a region with data, try to start with the first lineout where the data is already clearly visible. 
   
   If your fit ranges look good, the next thing is to check that the background was adequitely handled. If your fits dont match the data outside the EPW features then the background is likely wrong. 

   Folloing that the next most likely problem is the initial conditions and bounds, starting too far from the correct answer can result in the minimizer take steps that are too large for it to find the correct minimum. Try different starting conditions that are likely to be closer to the correct value (see the page :ref:`Fundamentals of Thomson Scattering <ts_fundamentals>` if you are not sure how to guess better values). Also try not to use bound that are very tight, if you are not certain in the plasma conditons having the bound to close to the inital condition can prevent the minimizer from reaching the correct value. 

   Lastly if all these other parameters seem good its probably a model selection problem. The input deck essentially creates a "model" for the plasma with a specific number of species with certain free parameters, this must line up with the actual plasma in order to produce high quality fits. For example an input deck with an ion species omited will not be able to reproduce the measured spectrum. 

   If you are struggling with your fits feel free to reach out to the developers for suggestions.