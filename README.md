

# smp\_graphs

Specifiy experiments as a graph of nodes and a set of signals,
corresponding nicely with tapping approach.

-   x load graph and execute in given order, each node knows which bus line maps onto its inputs

-   x reuse config as block in other config / nested experiments

-   x logging

-   input mapping: what are inputs / params, how to select from busses
    and map to local variable

-   input buffering: ring buffer decorator

-   blocksize vs. numsteps

-   read/write: ros

-   read/write: osc

-   sync vs. async nodes

-   functional, decorators

-   recurrence, backprop, signal propagation and execution order

-   networkx?


## Predecessors

My foruth attempt at a framework for computational sensorimotor
learning experiments. Earlier attempts are:

-   smp\_experiments: config as name value pairs and some python code
    wrapping to reuse singular experiments in an outer loop
-   smpblocks: first attempt at python config files containing a
    dictionary specifying generic computation blocks and their
    connections
-   smq: smpblocks had too small granularity, here i tried to be more
    specific introducing specific and fixed modules world, robot,
    brain. Alas it turned out that was too specific and didn't fit
    experiments deviating from that paradigm. Is where we are ;)

