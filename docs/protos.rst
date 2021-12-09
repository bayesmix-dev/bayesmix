bayesmix/protos

Protos
======

This library depends on Google's cross-platform Protocol Buffers library, also known as ``protobuf``, to store and move structured data.
This happens via classes known as ``Protobuf`` messages, or protos for short, which store information via serialization, i.e. compressing large portions of data into strings which can then be de-serialized back to normal form.
Skeletons for these objects are defined in ``.proto`` files, while the ``protoc`` compiler automatically builds the corresponding C++ and Python classes off of them.
This allows easy interface between multiple programming languages and *a posteriori* analysis of MCMC chains.

A description of all protos used in ``bayesmix`` follows.
They range from simple enumerator identifiers, vectors or matrices, to objects representing probability distributions, hyperpriors, states, or hyperparameter values.
Some of these protos are embedded in one another, possibly using the ``oneof`` keyword, which allows the outer proto to flexibly choose and contain one type of object among many different ones.
For instance, this is the case with protos representing hyperpriors, which can have increasing degrees of complexity depending on which model is chosen by the user.

.. raw:: html
    :file: protos.html
