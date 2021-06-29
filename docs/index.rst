.. bayesmix documentation master file, created by
   sphinx-quickstart on Sun Jun 27 08:35:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

bayesmix: a nonparametric C++ library for mixture models
============

.. image:: ../resources/logo_full.svg
   :width: 250px
   :alt: bayesmix full logo

.. image::
   https://readthedocs.org/projects/bayesmix/badge/?version=latest
   :target: https://bayesmix.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

``bayesmix`` is a C++ library for running MCMC simulations in Bayesian mixture models.
It uses the ``Eigen`` library for vector-matrix manipulation and linear algebra, and ``protobuf`` (Protocol Buffers) for communication and storage of structured data.



Submodules
==========

There are currently three submodules to the ``bayesmix`` library, represented by three classes of objects:

- ``Algorithms``
- ``Hierarchies``
- ``Mixings``.

.. toctree::
   :maxdepth: 1
   :caption: API: library submodules

   algorithms
   hierarchies
   mixings
   collectors
   utils



Tutorials
=========

TODO coming soon!




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
