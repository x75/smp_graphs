

# smp\_graphs

Specifiy sensorimotor learning experiments as a graph of nodes and a
set of signals, corresponding nicely with the tapping approach.


## Items

-   x dict printing for dynamic reconf inspection

-   x base block

-   x two-pass init: complete by putting input init into second pass

-   loop block
    -   x parallel loop within graph, modify graph
    -   sequential loop for running block variations e.g hyperopt or evo

-   logging
    -   x std logging OK
    -   include git revision, initial and final config in log

-   x networkx for visualization?
    -   x standalone networkx graph from final config
    -   x graphviz

-   step, blocksize, ibuf
    -   min blocksize after pass 1
    -   how to optimize of min bs > 1?

-   sync / async block execution

-   minor stuff
    -   x print\_dict print fully compilable python code


## Examples

    cd smp_graph/experiments
    python experiment.py --conf conf/default2.conf
    python experiment.py --conf conf/default2_loop.conf


# smp\_graphs legacy notes


## v1 stalled

-   x load graph and execute in given order, each node knows which bus line maps onto its inputs

-   x reuse config as block in other config / nested experiments

-   x logging

-   x input mapping: what are inputs / params, how to select from busses
    and map to local variable

-   x input buffering: ring buffer decorator: use np.roll hoping it
    does that internally

-   global config: e.g. fileblock setting its odim, use global config
    inside blocks, dynamic graph structure, change notification. make
    dynamic foo and write back the results into config
    -   loop block / dynamic blocks that can change the graph

-   file sources: load data in config or in block?

-   blocksize vs. numsteps

-   bus: make bus a structured dict, allowing blocks to have several
    outputs, this make odim obsolete

-   read/write: ros

-   read/write: osc

-   sync vs. async nodes

-   x functional, decorators

-   recurrence, backprop, signal propagation and execution order

-   networkx?


## Predecessors

My fourth attempt at a framework for computational sensorimotor
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

