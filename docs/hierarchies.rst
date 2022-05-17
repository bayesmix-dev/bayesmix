bayesmix/hierarchies

Hierarchies
===========

In our algorithms, we store a vector of hierarchies, each of which represent a parameter :math:`\theta_h`.
The hierarchy implements all the methods needed to update :math:`\theta_h`: sampling from the prior distribution :math:`P_0`, the full-conditional distribution (given the data {:math:`y_i` such that :math:`c_i = h`} ) and so on.

In BayesMix, each choice of :math:`G_0` is implemented in a different ``PriorModel`` object and each choice of :math:k(\cdot \mid \cdot)` in a ``Likelihood`` object, so that it is straightforward to create a new ``Hierarchy`` using one of the already implemented priors or likelihoods.
The sampling from the full conditional of :math:`\theta_h` is performed in an ``Updater`` class.
`State` classes  are used to store  parameters ``\theta_h`s of every mixture component.
Their main purpose is to handle serialization and de-serialization of the state

.. toctree::
   :maxdepth: 1
   :caption: API: hierarchies submodules

   likelihoods
   prior_models
   updaters
   states


-------------------------
Main operations performed
-------------------------

A hierarchy must be able to perform the following operations:

a. Sample from the prior distribution: generate :math:`\theta_h \sim P_0` [``sample_prior``]
b. Sample from the 'full conditional' distribution: generate theta_h from the distribution :math:`p(\theta_h \mid \cdots ) \propto P_0(\theta_h) \prod_{i: c_i = h} k(y_i | \theta_h)` [``sample_full_conditional``]
c. Update the hyperparameters involved in :math:`P_0` [``update_hypers``]
d. Evaluate the likelihood in one point, i.e. :math:`k(x | \theta_h)` for theta_h the current value of the parameters [``like_lpdf``]
e. When :math:`k` and :math:`P_0` are conjugate, we must also be able to compute the marginal/prior predictive distribution in one point, i.e. :math:`m(x) = \int k(x | \theta) P_0(d\theta)`, and the conditional predictive distribution :math:`m(x | \textbf{y} ) = \int k(x | \theta) P_0(d\theta | \{y_i: c_i = h\})` [``prior_pred_lpdf``, ``conditional_pred_lpdf``]

Moreover, the following utilities are needed:

f. write the current state :math:`\theta_h` into a appropriately defined Protobuf message [``write_state_to_proto``]
g. restore theta_h from a given Protobuf message [``set_state_from_proto``]
h. write the values of the hyperparameters in :math:`P_0` to a Protobuf message [``write_hypers_to_proto``]


In each hierarchy, we also keep track of which data points are allocated to the hierarchy.
For this purpose ``add_datum`` and ``remove_datum`` are employed.
Finally, the update involeved in the full-conditional, especially if :math:`P_0` and :math:`k` are conjugate an semi-conjugate can be performed efficiently from a set of sufficient statistics, hence when ``add_datum`` or ``remove_datum`` are invoked, the method ``update_summary_statistics`` is called.


.. _hierarchies-crtp:

--------------
Code structure
--------------

We employ a Curiously Recurring Template Pattern (CRTP) coupled with an abstract interface, similarly to the ``Mixing`` class.
The code thus composes of: a virtual class defining the API, a template base class that is the base for the CRTP and derived child classes that fully specialize the template arguments.
The class ``AbstractHierarchy`` defines the API, i.e. all the methods that need to be called from outside of a ``Hierarchy`` class.
A template class ``BaseHierarchy`` inherits from ``AbstractHierarchy`` and implements some of the necessary virtual methods, which need not be implemented by the child classes.

Instead, child classes must implement:

a. ``like_lpdf``: evaluates :math:`k(x | \theta_h)`
b. ``marg_lpdf``: evaluates m(x) given some parameters :math:`\theta_h` (could be both the hyperparameters in :math:`P_0` or the paramters given by the full conditionals)
c. ``draw``: samples from :math:`P_0` given the parameters
d. ``clear_summary_statistics``: clears all the summary statistics
e. ``update_hypers``: performs the update of parameters in :math:`P_0` given all the :math:`\theta_h` (passed as a vector of protobuf Messages)
f. ``initialize_state``: initializes the current :math:`\theta_h` given the hyperparameters in :math:`P_0`
g. ``initialize_hypers``: initializes the hyperparameters in :math:`P_0` given their hyperprior
h. ``update_summary_statistics``: updates the summary statistics when an observation is allocated or de-allocated from the hierarchy
i. ``get_posterior_parameters``: returns the paramters of the full conditional distribution **possible only when** :math:`P_0` **and** :math:`k` **are conjugate**
j. ``set_state_from_proto``
k. ``write_state_to_proto``
l. ``write_hypers_to_proto``

Note that not all of these members are declared virtual in ``AbstractHierarchy`` or ``BaseHierarchy``: this is because virtual members are only the ones that must be called from outside the ``Hierarchy``, the other ones are handled via CRTP. Not having them virtual saves a lot of lookups in the vtables.
The ``BaseHierarchy`` class takes 4 template parameters:

1. ``Derived`` must be the type of the child class (needed for the CRTP)
2. ``State`` is usually a struct representing :math:`\theta_h`
3. ``Hyperparams`` is usually a struct representing the parameters in :math:`P_0`
4. ``Prior`` must be a protobuf object encoding the prior parameters.

.. Finally, a ``ConjugateHierarchy`` takes care of the implementation of some methods that are specific to conjugate models.


----------------
Abstract Classes
----------------

.. doxygenclass:: AbstractHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: BaseHierarchy
   :project: bayesmix
   :members:

---------------------------------
Classes for Conjugate Hierarchies
---------------------------------

.. doxygenclass:: NNIGHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: NNWHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: LinRegUniHierarchy
   :project: bayesmix
   :members:

-------------------------------------
Classes for Non-Conjugate Hierarchies
-------------------------------------

.. doxygenclass:: NNxIGHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: LapNIGHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: FAHierarchy
   :project: bayesmix
   :members:
