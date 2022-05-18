bayesmix/hierarchies/updaters

Updaters
========

An ``Updater`` implements the machinery to provide a sampling from the full conditional distribution of a given hierarchy.

The only operation performed is ``draw`` that samples from the full conditional, either exactly or via Markov chain Monte Carlo.

.. doxygenclass:: AbstractUpdater
    :project: bayesmix
    :members:

--------------
Code Structure
--------------

We distinguish between semi-conjugate updaters and the metropolis-like updaters.


Semi Conjugate Updaters
-----------------------

A semi-conjugate updater can be used when the full conditional distribution has the same form of the prior. Therefore, to sample from the full conditional, it is sufficient to call the ``draw`` method of the prior, but with an updated set of hyperparameters.

The class ``SemiConjugateUpdater`` defines the API

.. doxygenclass:: SemiConjugateUpdater
    :project: bayesmix
    :members:

Classes inheriting from this one should only implement the ``compute_posterior_hypers(...)`` member function.


Metropolis-like Updaters
------------------------

A Metropolis updater uses the Metropolis-Hastings algorithm (or its variations) to sample from the full conditional density.

.. doxygenclass:: MetropolisUpdater
    :project: bayesmix
    :members:


Classes inheriting from this one should only implement the ``sample_proposal(...)`` method, which samples from the porposal distribution, and the ``proposal_lpdf`` one, which evaluates the proposal density log-probability density function.

---------------
Updater Classes
---------------

.. doxygenclass:: RandomWalkUpdater
    :project: bayesmix
    :members:
.. doxygenclass:: MalaUpdater
    :project: bayesmix
    :members:
.. doxygenclass:: NNIGUpdater
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: NNxIGUpdater
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: NNWUpdater
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: MNIGUpdater
    :project: bayesmix
    :members:
    :protected-members:
.. doxygenclass:: FAUpdater
    :project: bayesmix
    :members:
    :protected-members:
