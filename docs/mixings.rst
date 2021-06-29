bayesmix/mixings

Mixings
=======

In the algorithms of the library, we store a single ``Mixing`` object that represents the mixing measure used in the model.
There are two types of ``Mixing``s: marginal and conditional, each of which can only be used with the matching type of ``Algorithm``.
For both of these types, certain API functions are required.


--------------
Code structure
--------------

We employ a Curiously Recurring Template Pattern coupled with an abstract interface, similarly to the ``Mixing`` class. 
The code thus composes of: a virtual class defining the API, a template base class that is the base for the CRTP and derived child classes that fully specialize the template arguments.
The class ``AbstractMixing`` defines the API, i.e. all the methods that need to be called from outside of a ``Mixing`` class. 
A template class ``BaseMixing`` inherits from ``AbstractMixing`` and implements some of the necessary virtual methods, which need not be implemented by the child classes. 


-------
Classes
-------

.. doxygenclass:: AbstractMixing
   :project: bayesmix
   :members:
.. doxygenclass:: BaseMixing
   :project: bayesmix
   :members:
.. doxygenclass:: DirichletMixing
   :project: bayesmix
   :members:
.. doxygenclass:: PitYorMixing
   :project: bayesmix
   :members:
.. doxygenclass:: TruncatedSBMixing
   :project: bayesmix
   :members:
.. doxygenclass:: LogitSBMixing
   :project: bayesmix
   :members:
