

# smp\_graphs

Specifiy sensorimotor learning experiments as a graph of nodes and a
set of signals, corresponding nicely with the tapping approach. The
basis framework functions are independent of external libs but the
block implementations make use of other *smp* libs such as
[smp\_base](https://github.com/x75/smp_base). The design flow is similar to block based visual programming
approaches and DSP design techniques found in supercollider, puredata,
or gnuradio to name a few.


## Items

-   step, blocksize, ibuf
    -   x basic blocksize handling
    -   x check if logging still works properly
    -   x make prim blocks blocksize aware
    -   min blocksize after pass 1
    -   how to optimize if min(bs) > 1?

-   loop block
    -   x parallel loop within graph, modify graph
    -   sequential loop for running block variations e.g hyperopt or evo

-   sync / async block execution

-   read/write: ros, osc, &#x2026;

-   x logging
    -   x std logging OK
    -   x include git revision, initial and final config in log

-   x networkx for visualization?
    -   x standalone networkx graph from final config
    -   x graphviz

-   minor stuff
    -   x print\_dict print fully compilable python code
    -   x separate header/footer for full config file to remove code
        replication and clutter

-   x two-pass init: complete by putting input init into second pass

-   x base block

-   x dict printing for dynamic reconf inspection


## Examples

    cd smp_graph/experiments
    python experiment.py --conf conf/default2.conf
    python experiment.py --conf conf/default2_loop.conf


# Notes

This is my 5th attempt at designing a framework for computational
sensorimotor learning experiments. Earlier attempts include

-   **smp\_experiments**: defined config as name value pairs and some
    python code wrapping enabling the reuse of singular experiments
    defined elsewhere in an outer loop doing variations (collecting
    statistics, optimizing, &#x2026;)
-   **smpblocks**: first attempt at using plain python config files
    containing a dictionary specifying generic computation blocks and
    their connections. granularity was too small and specifying
    connections was too complicated
-   **smq**: tried to be more high-level, introducing three specific and
    fixed modules 'world', 'robot', 'brain'. Alas it turned out that
    left us too inflexible and obviosuly couldn't accomodate any
    experiments deviating from that schema. Is where we are ;)

