

# smp\_graphs

Specifiy sensorimotor learning experiments as a graph of nodes and a
set of signals, corresponding nicely with the tapping approach. The
basis framework functions are independent of external libs but the
block implementations make use of other *smp* libs such as
[smp\_base](https://github.com/x75/smp_base).

The design flow is similar to block based visual programming
approaches and DSP design techniques found in supercollider, puredata,
or gnuradio, to name a few. The idea is to use predefined blocks that
implement a function of the block's inputs. The assignment of values
to a given input is part of the graph configuration and is either a
constant computed at configuration time or another block's output
provided on a globally accessible bus structure. Every block writes
it's outputs to the bus.


## Framework items

-   kinesis rate param for blocks = blocksize?
-   can we map top down - bottom up flow nicely into the graph?

-   don't need to copy outputs of subgraph because the bus is global,
    FIXME consider making hierarchical bus identifiers or assert all
    keys and subkeys uniq

-   step, blocksize, ibuf
    -   min blocksize after pass 1
    -   how to optimize if min(bs) > 1?
    -   x make prim blocks blocksize aware
    -   x check if logging still works properly
    -   x basic blocksize handling

-   hierarchical composition
    -   two ways of handling subgraphs: 1) insert into flattened
        topgraph, 2) keep hierarchical graph structure, for now going
        with 1)
    -   think about these issues: outer vs. inner numsteps and blocksizes,
        how to get data in and out in a subgraph independent way
    -   for now: assert inner numsteps <= outer numsteps, could either
        enforce 1 or equality
    -   x use blocks that contain other graphs (default2\_hierarchical.py)

-   loop block
    -   special hierarchical block with additional spec about how often
        and with which variations to iterate the subgraph
    -   x sequential loop for running block variations e.g hyperopt or evo,
        for now assert blocksize = numloops, one loop iteration returns
        one data point
    -   x parallel loop within graph, modify graph. this is different
        from dynamic containment

-   sync / async block execution

-   read/write: integrate input from and output to ROS, OSC, &#x2026;

-   x logging
    -   x std logging OK
    -   x include git revision, initial and final config in log

-   x networkx for visualization?
    -   x standalone networkx graph from final config
    -   x graphviz

-   misc stuff
    -   x separate header/footer for full config file to remove code
        replication and clutter

-   x two-pass init: complete by putting input init into second pass

-   x base block

-   dict printing for dynamic reconf inspection
    -   fix OrderedDict in reconstructed config dicts
    -   x print\_dict print compilable python code?
    -   x basic formatted dict printing. issues: different needs in
        different contexts, runtime version vs. init version. disregard
        runtime version in logging and storage


## Examples

Currently depends on the following python libs

-   External: numpy, matplotlib, pandas, networkx, hyperopt
-   smp world: smp\_base

smp stuff is 'installed' via setting the PYTHONPATH to include the
relevant directories like

    export PYTHONPATH=/path/to/smp_base:/path/to/smp_graphs:$PYTHONPATH

then go into smp\_graphs/experiments directory where experiments are
run from

    cd smp_graph/experiments

Example configurations are 

    # default2.py, test most basic functionality with const and random blocks
    python experiment.py --conf conf/default2.py

    # default2_loop.py, test the graph modifying loop block
    python experiment.py --conf conf/default2_loop.py

    # default2_hierarchical.py, test hierarchical composition loading a subblock from
    #                             an existing configuration
    python experiment.py --conf conf/default2_hierarchical.py

    # default2_loop_seq.py, test dynamic loop instantiating the loopblock
    #                         for every loop iteration
    python experiment.py --conf conf/default2_loop_seq.py

and so on. Other configurations are puppy\_rp.py and
puppy\_rp\_blocksize.py which load a logfile and do analysis on that
data.

Two utilities for inspecting logged configurations and data are
provided in util\_logdump.py and util\_logplot.py


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

