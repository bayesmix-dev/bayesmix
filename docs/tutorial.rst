bayesmix/utils

Univariate data
===============

You can run the ``bash/tutorial.sh`` script for a quick example on how to use the ``run.cc`` file.
This example uses files which are available in the ``resources/tutorial`` subfolder.
We recommend that you open and read these files for a better understanding of how the process works.
The aforementioned script executes the following command:

.. code-block:: shell
  build/run \
    resources/tutorial/algo.asciipb \
    NNIG resources/tutorial/nnig_ngg.asciipb \
    DP   resources/tutorial/dp_gamma.asciipb \
    "" \
    resources/tutorial/data.csv \
    resources/tutorial/grid.csv \
    resources/tutorial/out/density.csv \
    resources/tutorial/out/numclust.csv \
    resources/tutorial/out/clustering.csv

In general, the pattern to use the executable is:

``build/run \
  algorithm_settings_file \
  hierarchy_type  hierarchy_prior_file \
  mixing_type  mixing_prior_file \
  collector_name \
  data_file \
  grid_file \
  density_output_file \
  numclust_output_file \
  clustering_output_file \
  [hierarchy_covariates_file] \
  [hierarchy_grid_covariates_file] \
  [mixing_covariates_file] \
  [mixing_grid_covariates_file]``

---------
Arguments
---------

The arguments provided to the executable are:
* ``algorithm_settings_file``: the text file that contains algorithm parameters, such as the actual algorithm to be used and the number of MCMC iterations
* ``hierarchy_type``: the identifier for the type of hierarchy to be used. In the above case we use a Normal-Normal-InverseGamma (NNIG) hierarchy
* ``hierarchy_prior_file``: the text file that contains information about the prior and hyperprior values for the hierarchy. In the above case, we use a Normal-Gamma-Gamma (NGG) hyperprior
* ``mixing_type``: the identifier for the type of mixing to be used. In the above case, we use a Dirichlet Process (DP) mixture
* ``mixing_prior_file``: the text file that contains information about the prior and hyperprior values for the mixing. In the above case, we use a Gamma hyperprior on the DP total mass parameter
* ``collector_name``: the name of the file 
* ``data_file``:
* ``grid_file``:
* ``density_output_file``:
* ``numclust_output_file``:
* ``clustering_output_file``:
* the remaining arguments ``hierarchy_covariates_file``, ``hierarchy_grid_covariates_file``, ``mixing_covariates_file``, and ``mixing_grid_covariates_file`` are optional, and are only to be used if the model is dependent on covariates, which is not the case for our NNIG + DP example.

------
Output
------

The output of the program should look something like this:

``Running run.cc
Initializing... Done
Running Neal3 algorithm with NNIG hierarchies, DP mixing...
[============================================================] 100% 0.157s
Done
Computing log-density...
[============================================================] 100% 0.045s
Done
Successfully wrote density to resources/tutorial/out/density.csv
Successfully wrote cluster sizes to resources/tutorial/out/numclust.csv
Computing cluster estimate...
(Computing mean dissimilarity... Done)
[============================================================] 100% 0.056s
Done
Successfully wrote clustering to resources/tutorial/out/clustering.csv
End of run.cc``

This means that the output has been written into the indicated files.
You can open them with your favorite CSV reader, or manipulate them with Python libraries in order to make plots, or whatever you wish.
