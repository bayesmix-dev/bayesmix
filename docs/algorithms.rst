bayesmix/algorithms

The ``Algorithm`` class handles other class objects and performs the MCMC simulation.
There are two types of ``Algorithm``: marginal and conditional, each of which can only be used with the matching type of ``Mixing``.

Algorithms
==========
.. doxygenclass:: BaseAlgorithm
   :project: bayesmix
   :members:
.. doxygenclass:: MarginalAlgorithm
   :project: bayesmix
   :members:
.. doxygenclass:: Neal2Algorithm
   :project: bayesmix
   :members:
.. doxygenclass:: Neal3Algorithm
   :project: bayesmix
   :members:
.. doxygenclass:: Neal8Algorithm
   :project: bayesmix
   :members:
.. doxygenclass:: SplitAndMergeAlgorithm
   :project: bayesmix
   :members:
.. doxygenclass:: ConditionalAlgorithm
   :project: bayesmix
   :members:
.. doxygenclass:: BlockedGibbsAlgorithm
   :project: bayesmix
   :members:
