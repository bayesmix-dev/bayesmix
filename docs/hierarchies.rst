bayesmix/hierarchies

Hierarchies
===========

In our algorithms, we store a vector of hierarchies, each of which represent a parameter theta_h.
The hierarchy implements all the methods needed to update theta_h: sampling from the prior distribution (P_0), the full conditional distribution (given the data {y_i such that c_i = h} ) and so on.


-------------------------
Main operations performed
-------------------------

A hierarchy must be able to perform the following operations

1. Sample from the prior distribution: generate theta_h ~ P_0 [`sample_prior`]
2. Sample from the 'full conditional' distribution: generate theta_h from the distribution



.. role:: raw-latex(raw)
        :format: latex html

    .. raw:: html

       <script type="text/javascript" src="http://localhost/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

    This: :raw-latex:`\((x+a)^3\)`

    this: :raw-latex:`\(W \approx \sum{f(x_k) \Delta x}\)`

    this: :raw-latex:`\(W = \int_{a}^{b}{f(x) dx}\)`

    and this:

    .. raw:: latex html

       \[ \frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} =
              1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}}
              {1+\frac{e^{-8\pi}} {1+\ldots} } } } \]



.. math::

   \\frac{ \\sum_{t=0}^{N}f(t,k) }{N}


or :math:`p(\\theta_h \\mid \\cdots ) \\propto P_0(\\theta_h) \\prod_{i: c_i = h} k(y_i | \\theta_h)` [`sample_full_conditional`]
3. Update the hyperparameters involved in P_0 [`update_hypers`]
4. Evaluate the likelihood in one point, i.e. k(x | \theta_h) for theta_h the current value of the parameters [`like_lpdf`]
5. When k and P_0 are conjugate, we must also be able to compute the marginal/prior predictive distribution in one point, i.e. 
<a href="https://www.codecogs.com/eqnedit.php?latex=m(x)&space;=&space;\int&space;k(x&space;|&space;\theta)&space;P_0(d\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m(x)&space;=&space;\int&space;k(x&space;|&space;\theta)&space;P_0(d\theta)" title="m(x) = \int k(x | \theta) P_0(d\theta)" /></a>
and the conditional predictive distribution: 
<a href="https://www.codecogs.com/eqnedit.php?latex=m(x&space;|&space;\textbf{y}&space;)&space;=&space;\int&space;k(x&space;|&space;\theta)&space;P_0(d\theta&space;|&space;\{y_i:&space;c_i&space;=&space;h\})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m(x&space;|&space;\textbf{y}&space;)&space;=&space;\int&space;k(x&space;|&space;\theta)&space;P_0(d\theta&space;|&space;\{y_i:&space;c_i&space;=&space;h\})" title="m(x | \textbf{y} ) = \int k(x | \theta) P_0(d\theta | \{y_i: c_i = h\})" /></a>
[`prior_pred_lpdf`, `conditional_pred_lpdf`]


Moreover, the following utilities are needed:

6. write the current state (theta_h) into a appropriately defined Protobuf message [`write_state_to_proto`]
7. restore theta_h from a given Protobuf message [`set_state_from_proto`]
8. write the values of the hyperparameters in P_0 to a Protobuf message [`write_hypers_to_proto`]


In each hierarchy, we also keep track of which data points are allocated to the hierarchy. 
For this purpose `add_datum` and `remove_datum` are employed.
Finally, the update involeved in the full_conditional, especially if P_0 and k are conjugate an semi-conjugate can be performed efficiently from a set of sufficient statistics, hence when `add_datum` or `remove_datum` are invoked, the method `update_summary_statistics` is called.


--------------
Code structure
--------------

We employ a Curiously Recurring Template Pattern coupled with an abstract interface. 
The code thus composes of: a virtual class defining the API, a template base class that is the base for the CRTP and derived child classes that fully specialize the template arguments.

The class `AbstractHierarchy` defines the API, i.e. all the methods that need to be called 
from outside of a Hierarchy class. 
A template class `BaseHierarchy` inherits from `AbstractHierarchy` and implements some of the virtual methods in in, specifically: `sample_prior`, `sample_full_cond`, `initialize`, `add_datum`, `remove_datum`, `prior_pred_lpdf`, `conditional_pred_ldpf` `get_mutable_prior`, `like_lpdf_grid`, `prior_pred_lpdf_grid` and `conditional_pred_lpdf_grid`.
These methods do not need to be implemented by the child classes. 

Instead, child classes must implement:

1. `like_lpdf`: evaluates k(x | theta_h)
2. `marg_lpdf`: evaluates m(x) given some parameters theta_h (could be both the hyperparameters in P_0 or the paramters given by the full conditionals)
3. `draw`: samples from P_0 given the parameters
4. `clear_data`: clears all the summary statistics
5. `update_hypers`: performs the update of parameters in P_0 given all the theta_h's (passed as a vector of protobuf Messages)
6. `initialize_state`: initializes the current theta_h given the hyperparameters in P_0
7. `initialize_hypers`: initializes the hyperparameters in P_0 given their hyperprior
8. `update_summary_statistics`: updates the summary statistics when an observation is allocated or de-allocated from the hierarchy
9. `get_posterior_parameters`: returns the paramters of the full conditional distribution **possible only when P_0 and k are conjugate**
10. `set_state_from_proto`
11. `write_state_to_proto`
12. `write_hypers_to_proto`


Observe that not all of these members are declared virtual in `AbstractHierarchy` or `BaseHierarchy`: this is because virtual members are only the ones that must be called from outside the `Hierarchy`, the other ones are handled via CRTP. Not having them virtual saves a lot of lookups in the vtables.

The BaseHierarchy class takes 4 template parameters:
1. `Derived` must be the type of the child class (needed for the CRTP)
2. `State` is usually a struct representing theta_h
3. `Hyperparams` is usually a struct representing the parameters in P_0
4. `Prior` must be a protobuf object encoding the prior parameters.


Finally, a ConjugateHierarchy takes care of the implementation of some methods that are specific to conjugate models.

-------
Classes
-------

.. doxygenclass:: AbstractHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: BaseHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: ConjugateHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: NNIGHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: NNWHierarchy
   :project: bayesmix
   :members:
.. doxygenclass:: LinRegUniHierarchy
   :project: bayesmix
   :members:
