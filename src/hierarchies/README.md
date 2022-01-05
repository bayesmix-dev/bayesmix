#Bayesian Hierarchical Models : a Curiously Recurring Template Pattern approach

## Overview: mixture models

Consider a mixture model of the kind

<a href="https://www.codecogs.com/eqnedit.php?latex=y_1,&space;\ldots,&space;y_n&space;\sim&space;\sum_{h=1}^M&space;w_h&space;k(\cdot&space;\mid&space;\theta_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_1,&space;\ldots,&space;y_n&space;\sim&space;\sum_{h=1}^M&space;w_h&space;k(\cdot&space;\mid&space;\theta_h)" title="y_1, \ldots, y_n \sim \sum_{h=1}^M w_h k(\cdot \mid \theta_h)" /></a>

coupled with a prior on the parameters:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_i&space;\sim&space;P_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_i&space;\sim&space;P_0" title="\theta_i \sim P_0" /></a>

the two quantities k (i.e. the component 'likelihood')  and P_0 define what we call a 'hierarchy'.
In our code, we use the terms 'hierarchy', 'unique value' and 'cluster' interchangeably when referring to the hierarchies.

This is a basic building block of all MCMC algorithms form mixture models. Moreover, P_0 and k cannot be handled separately, since the update of parameters theta_h depends on both k and P_0.

Moreover, P_0 likely depends on some parameters, and a prior can be placed on those.

The mixture model can be further rewritten as 

<a href="https://www.codecogs.com/eqnedit.php?latex=y_i&space;|&space;c_i=h&space;\sim&space;k(\cdot&space;|&space;\theta_h)&space;\\&space;c_1,&space;\ldots,&space;c_n&space;\sim&space;\text{Categorical}([M],&space;\textbf{w})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i&space;|&space;c_i=h&space;\sim&space;k(\cdot&space;|&space;\theta_h)&space;\\&space;c_1,&space;\ldots,&space;c_n&space;\sim&space;\text{Categorical}([M],&space;\textbf{w})" title="y_i | c_i=h \sim k(\cdot | \theta_h) \\ c_1, \ldots, c_n \sim \text{Categorical}([M], \textbf{w})" /></a>

In our algorithms, we store a vector of hierarchies, each of which represent a parameter theta_h.
The hierarchy implements all the methods needed to update theta_h: sampling from the prior distribution (P_0), the full conditional distribution (given the data {y_i such that c_i = h} ) and so on.


## Main operations performed

Therefore, a hierarchy must be able to perform the following operations

1. Sample from the prior distribution: generate theta_h ~ P_0 [`sample_prior`]
2. Sample from the 'full conditional' distribution: generate theta_h from the distribution 
p(theta_h)  into<a href="https://www.codecogs.com/eqnedit.php?latex=p(\theta_h&space;\mid&space;\cdots&space;)&space;\propto&space;P_0(\theta_h)&space;\prod_{i:&space;c_i&space;=&space;h}&space;k(y_i&space;|&space;\theta_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\theta_h&space;\mid&space;\cdots&space;)&space;\propto&space;P_0(\theta_h)&space;\prod_{i:&space;c_i&space;=&space;h}&space;k(y_i&space;|&space;\theta_h)" title="p(\theta_h \mid \cdots ) \propto P_0(\theta_h) \prod_{i: c_i = h} k(y_i | \theta_h)" /></a>
[`sample_full_conditional`]
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


## Code structure

We employ a Curiously Recurring Template Pattern coupled with an abstract interface. 
The code thus composes of: a virtual class defining the API, a template base class that is the base for the CRTP and derived child classes that fully specialize the template arguments.

The class `AbstractHierarchy` defines the API, i.e. all the methods that need to be called from outside of a Hierarchy class.
A template class `BaseHierarchy` inherits from `AbstractHierarchy` and implements some of the virtual methods in in, specifically: `sample_prior`, `sample_full_cond`, `initialize`, `add_datum`, `remove_datum`, `prior_pred_lpdf`, `conditional_pred_ldpf` `get_mutable_prior`, `like_lpdf_grid`, `prior_pred_lpdf_grid` and `conditional_pred_lpdf_grid`.
These methods do not need to be implemented by the child classes. 

Instead, child classes must implement:

1. `like_lpdf`: evaluates k(x \| theta_h)
2. `marg_lpdf`: evaluates m(x) given some parameters theta_h (could be both the hyperparameters in P_0 or the paramters given by the full conditionals)
3. `draw`: samples from P_0 given the parameters
4. `clear_summary_statistics`
5. `update_hypers`: performs the update of parameters in P_0 given all the theta_h's (passed as a vector of protobuf Messages)
6. `initialize_state`: initializes the current theta_h given the hyperparameters in P_0
7. `initialize_hypers`: initializes the hyperparameters in P_0 given their hyperprior
8. `update_summary_statistics`: updates the summary statistics when an observation is allocated or de-allocated from the hierarchy
9. `compute_posterior_hypers`: returns the paramters of the full conditional distribution, which is **only possible when P_0 and k are conjugate**
10. `set_state_from_proto`
11. `write_state_to_proto`
12. `write_hypers_to_proto`

Observe that not all of these members are declared virtual in AbstractHierarchy or BaseHierarchy: this is because virtual members are only the ones that must be called from outside the Hierarchy, the other ones are handled via CRTP. Not having them virtual saves a lot of lookups in the vtables.

The BaseHierarchy class takes 4 template parameters:
1. `Derived` must be the type of the child class (needed for the CRTP)
2. `State` is usually a struct representing theta_h
3. `Hyperparams` is usually a struct representing the parameters in P_0
4. `Prior` must be a protobuf object encoding the prior parameters.

Finally, a `ConjugateHierarchy` takes care of the implementation of some methods that are specific to conjugate models.
