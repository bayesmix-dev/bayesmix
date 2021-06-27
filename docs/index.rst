.. bayesmix documentation master file, created by
   sphinx-quickstart on Sun Jun 27 08:35:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bayesmix: a nonparametric C++ library for mixture models
========================================================

.. image:: ../resources/logo_full.svg
   :width: 250px
   :alt: bayesmix full logo

.. image::
   https://readthedocs.org/projects/bayesmix/badge/?version=latest
   :target: https://bayesmix.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

``bayesmix`` is a C++ library for running MCMC simulations in Bayesian mixture models.

Current state of the software:

- ``bayesmix`` performs inference for mixture models with the following form:

.. math::
   y_1, \ldots, y_n \sim \int k(\cdot \mid \theta) \Pi(d\theta) \\
   \Pi \sim P

Where P is either the Dirichlet process or the Pitman-Yor process.

- We currently support univariate and multivariate location-scale mixture of Gaussian densities

- Inference is carried out using algorithms such as Neal's Algorithm 2 from `Neal (2000)`_.
.. _Neal (2000): http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf

- Serialization of the MCMC chains is possible using Google's `Protocol Buffers library`_.
.. _Protocol Buffers library: https://developers.google.com/protocol-buffers


Submodules
==========
There are currently three submodules to the ``bayesmix`` library, playing the following roles:

- ``Algorithms``;
- ``Hierarchies``;
- ``Mixings``.


.. toctree::
   :maxdepth: 1
   :caption: API: library submodules

   algorithms
   hierarchies
   mixings
   collectors
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
