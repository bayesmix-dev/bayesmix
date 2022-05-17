bayesmix/hierarchies/likelihoods/states

States
======

``States`` are classes used to store  parameters :math:`\theta_h` of every mixture component.
Their main purpose is to handle serialization and de-serialization of the state.
Moreover, they allow to go from the constrained to the unconstrained representation of the parameters (and viceversa) and compute the associated determinant of the Jacobian appearing in the change of density formula.


--------------
Code Structure
--------------

All classes must inherit from the `BaseState` class

.. doxygenclass:: State::BaseState
    :project: bayesmix
    :members:

Depending on the chosen ``Updater``, the unconstrained representation might not be needed, and the methods ``get_unconstrained()``, ``set_from_unconstrained()`` and ``log_det_jac()`` might never be called.
Therefore, we do not force users to implement them.
Instead, the ``set_from_proto()`` and ``get_as_proto()`` are fundamental as they allow the interaction with Google's Protocol Buffers library.

-------------
State Classes
-------------

.. doxygenclass:: State::UniLS
    :project: bayesmix
    :members:

.. doxygenclass:: State::MultiLS
    :project: bayesmix
    :members:

.. doxygenclass:: State::FA
    :project: bayesmix
    :members:
    :protected-members:
