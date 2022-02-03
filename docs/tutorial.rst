bayesmix/utils

Tutorial - Univariate data
==========================

You can run the ``examples/tutorial/run.sh`` shell script for a quick example on how to use the ``run_mcmc.cc`` file.
This example uses files which are available in the ``resources/tutorial`` subfolder.
We recommend that you check out these these files for a better understanding of how the process works.
The aforementioned script executes the following command:

.. code-block:: shell

  build/run_mcmc \
    --algo-params-file resources/tutorial/algo.asciipb \
    --hier-type NNIG --hier-args resources/tutorial/nnig_ngg.asciipb \
    --mix-type DP --mix-args resources/tutorial/dp_gamma.asciipb \
    --coll-name resources/tutorial/out/chains.recordio \
    --data-file resources/tutorial/data.csv \
    --grid-file resources/tutorial/grid.csv \
    --dens-file resources/tutorial/out/density.csv \
    --n-cl-file resources/tutorial/out/numclust.csv \
    --clus-file resources/tutorial/out/clustering.csv


The appropriate syntax for an argument is ``--arg-name`` followed by ``ARGVALUE``.
These arguments can be provided in any order, as long as they are preceded by the appropriate argument name.
Due to the large number of parameters, we recommend that such a command is written to a ``.sh`` script which is then executed, just like with the given example.
Don't forget to grant a new script execute permissions via ``chmod +x myscript.sh``!



---------
Arguments
---------

The following arguments are **mandatory**, i.e. they must be provided to the ``build/run_mcmc`` executable regardless of the type of run which is being performed.
Please refer to the corresponding sections of the :ref:`protos` page (or equivalently to the file with that name in the ``proto/`` subfolder) for more information:

* ``--algo-params-file``: the text file that contains algorithm parameters, such as the actual algorithm to be used and the number of MCMC iterations (see ``algorithm_params.proto``)
* ``--hier-type``: the identifier for the type of hierarchy to be used (see ``hierarchy_id.proto``). In the above case we use a Normal-Normal-InverseGamma (NNIG) hierarchy
* ``--hier-args``: the text file that contains information about the prior and hyperprior values for the hierarchy (see ``hierarchy_prior.proto``). In the above example, we use a Normal-Gamma-Gamma (NGG) hyperprior
* ``--mix-type``: the identifier for the type of mixing to be used (see ``mixing_id.proto``). In the above example, we use a Dirichlet Process (DP) mixture
* ``--mix-args``: the text file that contains information about the prior and hyperprior values for the mixing (see ``mixing_prior.proto``). In the above example, we use a Gamma hyperprior on the DP total mass parameter
* ``--coll-name``: the name of the ``FileCollector`` file in which to store the full information of the MCMC chain, in binary form. If you don't need to store it into a file, you can use the ``memory`` keywod instead, which will instruct the code to use a ``MemoryCollector``
* ``--data-file``: the input CSV file with the model data, where each row corresponds to one data point

The following arguments are optional, but if either is not provided, *density estimation* will not take place:

* ``--grid-file``: the input CSV file with the data points on which the log-predictive density will be evaluated. Again, each row corresponds to one grid point
* ``--dens-file``: the output CSV file in which the density estimation will be stored, in the form of a ``num_iterations x grid_size`` table

The following arguments are also optional, and involve *clustering information*:

* ``--n-cl-file``: the output CSV file in which the number of clusters at each iteration will be written, one per line
* ``--clus-file``: the output CSV file in which the allocation labels of the MCMC chain will be stored, in the form of a ``num_iterations x data_size`` table
* ``--best-clus-file``: the output CSV file in which the program will store the labels of the posterior clustering. This will be computed as the visited partition that minimizes the Binder loss function, which will take some further time to be computed

Finally, the remaining arguments are only to be used if the *model is dependent* on covariates, which is not the case for our NNIG + DP example:

* ``--hier-cov-file``: the input CSV file with the data covariates of the dependent hierarchy used (if any)
* ``--mix-cov-file``: the input CSV file with the data covariates of the dependent mixing used (if any)
* ``--hier-grid-cov-file`` (optional): the input CSV file with the grid covariates of the dependent hierarchy used (if any)
* ``--mix-grid-cov-file`` (optional): the input CSV file with the grid covariates of the dependent mixing used (if any)



------
Output
------

The output of the program should look something like this:

.. code-block:: shell

  Running run_mcmc.cc
  Creating FileCollector, writing to file: resources/tutorial/out/chains.recordio
  Initializing... Done
  Running Neal3 algorithm with NNIG hierarchies, DP mixing...
  [============================================================] 100% 3.655s
  Done
  Computing log-density...
  [============================================================] 100% 1.164s
  Done
  Successfully wrote density to resources/tutorial/out/density.csv
  Successfully wrote number of clusters to resources/tutorial/out/numclust.csv
  Successfully wrote cluster allocations to resources/tutorial/out/clustering.csv
  End of run_mcmc.cc

This means that the output has been written into the indicated files.
You can manipulate them with a CSV reader, Python library, etc in order to make plots or anything else you'd like.
