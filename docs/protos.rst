bayesmix/protos

Protos
======

This library depends on Google's `Protocol Buffers <https://developers.google.com/protocol-buffers>`_, also known as ``protobuf``, which provides a convenient way to define classes that represent structured data.
Special classes henceforth referred to as ``protobuf`` messages, or protos for short, can be defined in ``.proto`` files. A special compiler, ``protoc``, is automatically called by the library to generate C++ and/or Python classes for each message.
The ``protobuf`` runtime library provides fast serialization of messages into bytes, which can be used to save objects to disk or pass serialized objects from one language to another.

A description of all protos used in ``bayesmix`` follows.
These range from simple enumerator identifiers (enums) and basic data types such as vectors or matrices, to objects representing probability distributions, hyperpriors, states, or hyperparameter values.
Some of these protos are embedded in one another, possibly using the ``oneof`` keyword, which allows the outer proto to flexibly choose and contain one type of object among many different ones.
For instance, this is the case with protos representing hyperpriors, which can have increasing degrees of complexity depending on which model is chosen by the user.

The use of protos allows easy interface between multiple programming languages, as well as *a posteriori* analysis of MCMC chains.

.. raw:: html
    :file: protos.html
