bayesmix/hierarchies/likelihoods

Likelihoods
===========

The ``Likelihood`` sub-module represents the likelihood we have assumed for the data in a given cluster. Each ``Likelihood`` class represents the sampling model

.. math::
    y_1, \ldots, y_k \mid \bm{\tau} \stackrel{\small\mathrm{iid}}{\sim} f(\cdot \mid \bm{\tau})

for a specific choice of the probability density function :math:`f`.

-------------------------
Main operations performed
-------------------------

A likelihood object must be able to perform the following operations:

a. First of all, we require the ``lpdf()`` and ``lpdf\_grid()`` methods, which simply evaluate the loglikelihood in a given point or in a grid of points (also in case of a \emph{dependent} likelihood, i.e., with covariates associated to each observation) [``lpdf()`` and ``lpdf_grid``]
b. In case you want to rely on a Metropolis-like updater, the likelihood needs to evaluation of the likelihood of the whole cluster starting from the vector of unconstrained parameters [``cluster_lpdf_from_unconstrained()``]. Observe that the ``AbstractLikelihood`` class provides two such methods, one returning a ``double`` and one returning a ``stan::math::var``. The latter is used to automatically compute the gradient of the likelihood via Stan's automatic differentiation, if needed. In practice, users do not need to implement both methods separately and can implement only one templated method
c. manage the insertion and deletion of a datum in the cluster [``add_datum`` and ``remove_datum``]
d. update the summary statistics associated to the likelihood [``update_summary_statistics``]. Summary statistics (when available) are used to evaluate the likelihood function on the whole cluster, as well as to perform the posterior updates of :math:`\bm{\tau}`. This usually gives a substantial speedup

--------------
Code structure
--------------

In principle, the ``Likelihood`` classes are responsible only of evaluating the log-likelihood function given a specific choice of parameters :math:`\bm{\tau}`.
Therefore, a simple inheritance structure would seem appropriate. However, the nature of the parameters :math:`\bm{\tau}` can be very different across different models (think for instance of the difference between the univariate normal and the multivariate normal paramters). As such, we employ CRTP to manage the polymorphic nature of ``Likelihood`` classes.

The class ``AbstractLikelihood`` defines the API, i.e. all the methods that need to be called from outside of a ``Likelihood`` class.
A template class ``BaseLikelihood`` inherits from ``AbstractLikelihood`` and implements some of the necessary virtual methods, which need not be implemented by the child classes.

Instead, child classes **must** implement:

a. ``compute_lpdf``: evaluates :math:`k(x \mid \theta_h)`
b. ``update_sum_stats``: updates the summary statistics when an observation is allocated or de-allocated from the hierarchy
c. ``clear_summary_statistics``: clears all the summary statistics
d. ``is_dependent``: defines if the given likelihood depends on covariates
e. ``is_multivariate``: defines if the given likelihood is for multivariate data

In case the likelihood needs to be used in a Metropolis-like updater, child classes **should** also implement:

f. ``cluster_lpdf_from_unconstrained``: evaluates :math:`\prod_{i: c_i = h} k(x_i \mid \tilde{\theta}_h)`, where :math:`\tilde{\theta}_h` is the vector of unconstrained parameters.

----------------
Abstract Classes
----------------

.. doxygenclass:: AbstractLikelihood
    :project: bayesmix
    :members:
.. doxygenclass:: BaseLikelihood
    :project: bayesmix
    :members:

----------------------------------
Classes for Univariate Likelihoods
----------------------------------

.. doxygenclass:: UniNormLikelihood
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: UniLinRegLikelihood
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: LaplaceLikelihood
    :project: bayesmix
    :members:
    :protected-members:

------------------------------------
Classes for Multivariate Likelihoods
------------------------------------

.. doxygenclass:: MultiNormLikelihood
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: FALikelihood
    :project: bayesmix
    :members:
    :protected-members:
