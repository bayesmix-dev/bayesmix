bayesmix/utils

Univariate data
===============

You can run the ``bash/tutorial.sh`` script for a quick example on how to use the ``run.cc`` file.
The aforementioned script executes the following command:
``
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
``
The arguments provided to the ``build/run`` executable are:

* sa
* sa
* prova

The output of the program will look something like this:
``
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
``
