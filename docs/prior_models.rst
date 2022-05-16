bayesmix/hierarchies/prior_models

Prior Models
============

A ``PriorModel`` represents the prior for the parameters in the likelihood,  i.e.

.. math::
    \bm{\tau} \sim G_{0}

with :math:`G_{0}` being a suitable prior on the parameters space. We also allow for more flexible priors adding further level of randomness (i.e. the hyperprior) on the parameter characterizing :math:`G_{0}`

-------------------------
Main operations performed
-------------------------

A likelihood object must be able to perform the following operations:

a. First of all, ``lpdf()`` and ``lpdf_from_unconstrained()`` methods evaluate the log-prior density function at the current state :math:`\bm \tau` or its unconstrained representation.
In particular, ``lpdf_from_unconstrained()`` is needed by Metropolis-like updaters.

b. The ``sample()`` method generates a draw from the prior distribution. If ``hier_hypers`` is ``nullptr, the prior hyperparameter values are used.
To allow sampling from the full conditional distribution in case of semi-congugate hierarchies, we introduce the ``hier_hypers`` parameter, which is a pointer to a ``Protobuf`` message storing the hierarchy hyperaprameters to use for the sampling.

c. The ``update_hypers()`` method updates the prior hyperparameters, given the vector of all cluster states.


--------------
Code structure
--------------

As for the ``Likelihood`` classes we employ the Curiously Recurring Template Pattern to manage the polymorphic nature of ``PriorModel`` classes.

The class ``AbstractPriorModel`` defines the API, i.e. all the methods that need to be called from outside of a ``PrioModel`` class.
A template class ``BasePriorModel`` inherits from ``AbstractPriorModel`` and implements some of the necessary virtual methods, which need not be implemented by the child classes.

Instead, child classes **must** implement:

a. ``lpdf``: evaluates :math:`G_0(\theta_h)`
b. ``sample``: samples from :math:`G_0` given a hyperparameters (passed as a pointer). If ``hier_hypers`` is ``nullptr``, the prior hyperparameter values are used.
c. ``set_hypers_from_proto``: sets the hyperparameters from a ``Probuf`` message
d. ``get_hypers_proto``: returns the hyperparameters as a ``Probuf`` message
e. ``initialize_hypers``: provides a default initialization of hyperparameters

In case you want to use a Metropolis-like updater, child classes **should** also implement:

f. ``lpdf_from_unconstrained``: evaluates :math:`G_0(\tilde{\theta}_h)`, where :math:`\tilde{\theta}_h` is the vector of unconstrained parameters.

----------------
Abstract Classes
----------------

.. doxygenclass:: AbstractPriorModel
    :project: bayesmix
    :members:
.. doxygenclass:: BasePriorModel
    :project: bayesmix
    :members:

--------------------
Non-abstract Classes
--------------------

.. doxygenclass:: NIGPriorModel
    :project: bayesmix
    :members:
    :protected-members:

.. doxygenclass:: NxIGPriorModel
    :project: bayesmix
    :members:
    :protected-members:

.. doxygenclass:: NWPriorModel
    :project: bayesmix
    :members:
    :protected-members:

.. doxygenclass:: MNIGPriorModel
    :project: bayesmix
    :members:
    :protected-members:

.. doxygenclass:: FAPriorModel
    :project: bayesmix
    :members:
    :protected-members:
