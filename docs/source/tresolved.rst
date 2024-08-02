Example: Fitting time-resolved data
-------------------------

This is an example for fitting time-resolved data. 



Running the code
^^^^^^^^^^^^^^^^^
Once you have adjusted the parameters, and saved the changes made. You will want to implement the run command.

.. code-block:: python

    python run_tsadar.py --cfg <path>/<to>/<inputs>/<folder> --mode fit

This command will yeild the following output, indicating the the fit is completed:

The following command will allow you to visualize the results of the fitting. The output link will redirect you to a local site where the outputs can be viewed. 
 
.. code-block:: shell

    mlflow ui

.. image:: _elfolder\mlflow_ui.png

Click the follow the link to vizialize the data. The resulting plots can be founs in the :bdg:`Artifacts` unedr the folder :bdg:`plots`. 
Best and worst folders contain the best and worst fits respectively. `



Fitting a new data set 
^^^^^^^^^^^^^^^^^^^^^^^
For fitting a new data set, it is recomended to fit a small region of the data using a small number of
linouts. This can be acomplished by setting the lineout:start and lineout:end to be close, or by increasing 
lineouts:skip
Once you have adjusted the inputs and outputs 


.. grid:: 2

    .. grid-item-card::  Inputs.yalm
        :link: inputs
        :link-type: ref

        Primary input deck 

    .. grid-item-card::  Defaults.yalm
        :link: configuring-the-default
        :link-type: ref

        Secondary input deck 

