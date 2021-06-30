bayesmix/utils

Tutorial - Univariate data
==========================

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

.. code-block:: shell

  build/run \
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
    [mixing_grid_covariates_file]


Due to the large number of parameters, it is recommended that such a command is written to a ``.sh`` script which is then executed, just like with the given example.



---------
Arguments
---------

The arguments which must be provided to the ``build/run`` executable are:

* ``algorithm_settings_file``: the text file that contains algorithm parameters, such as the actual algorithm to be used and the number of MCMC iterations
* ``hierarchy_type``: the identifier for the type of hierarchy to be used. In the above case we use a Normal-Normal-InverseGamma (NNIG) hierarchy
* ``hierarchy_prior_file``: the text file that contains information about the prior and hyperprior values for the hierarchy. In the above example, we use a Normal-Gamma-Gamma (NGG) hyperprior
* ``mixing_type``: the identifier for the type of mixing to be used. In the above example, we use a Dirichlet Process (DP) mixture
* ``mixing_prior_file``: the text file that contains information about the prior and hyperprior values for the mixing. In the above example, we use a Gamma hyperprior on the DP total mass parameter
* ``collector_name``: the name of the ``FileCollector`` file in which to store the full information of the MCMC chain, in binary form. If you don't need to store it into a file, you can use an empty string, which will instruct the code to use a ``MemoryCollector``. This is what we use in the above example
* ``data_file``: the input CSV file with the model data
* ``grid_file``: the input CSV file with the data points on which the density estimation will be evaluated
* ``density_output_file``: the output CSV file in which the density estimation will be stored, which will be a ``num_iterations x grid_size`` table
* ``numclust_output_file``: the output CSV file in which the number of clusters at each iteration will be written
* ``clustering_output_file``: the output CSV file in which the labels of the posterior clustering will be written
* the remaining arguments ``hierarchy_covariates_file``, ``hierarchy_grid_covariates_file``, ``mixing_covariates_file``, and ``mixing_grid_covariates_file`` are optional, and are only to be used if the model is dependent on covariates, which is not the case for our NNIG + DP example.



------
Output
------

The output of the program should look something like this:

.. code-block:: shell

  Running run.cc
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
  End of run.cc

This means that the output has been written into the indicated files.
You can open them with your favorite CSV reader, or manipulate them with Python libraries in order to make plots, or whatever you wish.
