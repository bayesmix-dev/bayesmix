==========================================
BayesMixPy: a Python interface to BayesMix
==========================================

Installation
============

After you have cloned the bayesmix github directory, navigate to the Python subfolder and install bayesmixpy using pip

.. code-block:: shell

    cd python
    pip3 install -e .


Usage
=====

`bayesmixpy` provides two functions: `build_bayesmix` and `run_mcmc`. The first one
installs `bayesmix` and its executables for you, while the second one calls the
executable that runs the MCMC sampler from Python.

Building bayesmix
-----------------

To build `bayesmix`, in a Python shell or a notebook write

.. code-block:: python

  from bayesmixpy import build_bayesmix

  n_proc = 4 # number of processors for building in parallel
  build_bayesmix(n_proc)


this will print out the installation log and, if the installation was successful, the following message

.. code-block:: shell

  Bayesmix executable is in '<BAYESMIX_HOME_REPO>/build',
  export the environment variable BAYESMIX_EXE=<BAYESMIX_HOME_REPO>build/run


Hence, for running the MCMC chain you should export the `BAYESMIX_EXE` environment variable. This can be done once and for all by copying

.. code-block:: shell 

  BAYESMIX_EXE=<BAYESMIX_HOME_REPO>build/run

in your .bashrc file (or .zshrc if you are a MacOs user), or every time you use bayesmixpy, you can add the following lines on top of your Python script/notebook

.. code-block:: python

  import os
  os.environ["BAYESMIX_EXE"] = <BAYESMIX_HOME_REPO>build/run

  from bayesmixpy import run_mcmc
  ....


Running bayesmix
----------------

To `run_mcmc` users must define the model and the algorithm in some configuration files or text strings. See the notebooks in `python/notebooks/gaussian_mix_uni.ipynb` and  `python/notebooks/gaussian_mix_multi.ipynb` for a concrete usage example.



The BayesmixPy Package
=========================


Functions
---------

.. automodule:: bayesmixpy
   :members:
   :undoc-members:
   :show-inheritance:
