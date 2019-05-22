"""The smp_graphs package is a framework designed for quick and
systematic creation, modification and execution of computational
experiments that involve dynamic closed-loop processes, mostly
learning agents and robots, reinforcement learning, etc. The intention
is to enable and facilitate fast reuse of functions, components, and
design patterns within this particular domain of computational
modelling for rapid prototyping and scientific
experimentation. Framework alternatively means `little
language <http://legacy.python.org/workshops/1998-11/proceedings/papers/aycock-little/aycock-little.html>`_,
`scientific workflow system <https://en.wikipedia.org/wiki/Scientific_workflow_system>`_,
*brain lego*, etc. The approach consists in strict separation of
function implementation and experiment description. Experiments are
described as graphs whose nodes are existing functional
blocks. Different types of interaction among the blocks are modelled
as edges. The graphs can also be nested which means that a node in a
graph at one level can either be a *primitive* function or a
*composite* function implemented as a graph itself.

Block basics
------------

Blocks are the fundamental functional building blocks and define the
computation of each node in the graph. Blocks are instantiated using a
configuration dictionary :data:`conf`. The items of the configuration
dict are copied into the Block's attributes on initialization. Blocks
need to provide a :func:`step` member function which is called at
every time step. Blocks come as *primitive* blocks, which provide
their own :func:`step` method and *composite* blocks which contain
subgraphs. Subgraphs can be specified explicitly as an OrderedDict or
a standard dict, or implicitly by loading an entire separate
configuration file from the configured path. See subgraphs_ section.

Common block attributes and, by correspondence, configuration items
are shown below in a pseudo configuration notation.

.. code:: python

    (
        'blockid',
        {
            'block': Block2,
            'params': {
                'id': 'blockid,
                'debug': False,
                'blocksize': int,
                'blockphase': list, 
               'inputs': {
                    'input-key-1': {'val': np.ndarray,          'shape': (idim-1, input-bufsize-1)},
                    'input-key-2': {'bus': 'nodeid/output-key', 'shape': (idim-busref, input-bufsize-2)},
                    # ...
                },
                'outputs': {
                    'output-key-1': {'shape': (odim-1, input-bufsize-1)},
                    'output-key-2': {'shape': (odim-2, input-bufsize-2)},
                    # ...
                },
            }
        }
    )


An experiment consists of a top block which is composite and contains
the top-level nxgraph which is constructed from the configuration file
provided on the command line. The config file is a plain python
file. During graph initialization the file is loaded, possibly
modified on the text level, compiled, possibly modified again at the
:mod:`ast` level and returned as a full runtime configuration
dict. The dict is used to construct a :mod:`networkx.Graph` whose
nodes' data includes the node block's class attribute in the 'block'
item, its configuration in 'params' item and the resulting live block
instance in the 'block_' item.

The top block's :func:`step` is called :data:`numsteps` times by the
:mod:`smp_graphs.experiment` shell. A composite block's step function,
which is the case for the top block, just iterates its subgraph and
calls each node's step function. Blocks schedule themselves for
execution with the :data:`blocksize` and :data:`blockphase`
attributes. The current :data:`cnt` is taken modulo the blocksize and
step is actually executed if the result is in the blockphase list of
integers. The default configuration is `{'blocksize': 1, 'blockphase':
[0]}` resulting in single time step execution.

The top block also has a :class:`smp_graphs.block.Bus` member which is a dictionary
whose keys are 'buskeys' and whose values are np.ndarrays and which is
globally available to all blocks. By convention, the block's
instantaneous outputs are computed by simply looping over the outputs
dict and copying the block attribute `block.output-key` to the bus
using the `buskey = 'block-id/output-key'. Block inputs can be
directly provided in the configuration as np.ndarrays when they are
constant, but more interestingly, the inputs can be assigned from any
bus item, including the block's own outputs with the 'bus': 'buskey'
config entry.


.. note::

    The bus item values can be multidimensional arrays (tensors) allowing to pass around complex data types like data batches, images, and so on. By convention, the last dimension of a bus item refers to time. This allows blocks to process input chunks and produce output chunks which have shapes different from their own blocksize.


.. note::

    This model has some consequences that might need to be considered when designing a graph. smp graphs are ordered dictionaries and this order is the verbatim execution order of the nodes. This means that a block's inputs, which refer to a block's output further down the graph order, will be dalyed by one time step. This is also the case for self-feedback connections.

Loop blocks: besides the basic composite blocks containing a subgraph
specification there are two special composite blocks which have
special powers to modify the graph itself. The :class:`LoopBlock2`
does so at initialization time and the :class:`SeqLoopBlock2` does so
at run time. The configuration mechanism is the same for both. Each
takes its subgraph (historically called 'loopblock') and applies
variable substitutions to the subgraph configuration using the `loop`
attribute which can be a list of tuples like `[('config-key',
'config-value'), ...]` or a pointer to a function yielding such tuples
on each consecutive call.

The set of standard blocks includes :class:`FuncBlock2`,
:class:`LoopBlock2`, :class:`SeqLoopBlock2`, :class:`PrimBlock2`,
:class:`IBlock2`, :class:`dBlock2`, :class:`DelayBlock2`,
:class:`SliceBlock2`, :class:`StackBlock2`, :class:`ConstBlock2`,
:class:`CountBlock2`, :class:`UniformRandomBlock2`

There is a growing set of fancy blocks for doing complex I/O, reading
file input, talking to realtime systems, numerical multivariate
analysis, and plotting.

.. seealso::

 - the experiments/conf/example_*.py files each demonstrate in a
   minimal fashion most of the basic features.

.. _subgraphs::
subgraphs
---------

Graphs and subgraphs in composite blocks allow reuse of other graphs
within the current graph. For full flexibility, parts of included
subgraphs need to be reconfigured from the current scope. This is not
finally solved yet, pending consolidation of graph, subgraph, and
loopblock. Currently there are several mechanisms for reconfiguration:

1-Block2 / subgraphconf
+++++++++++++++++++++++

Use a Block2 with 'subgraph' and 'subgraphconf' configuration entries

.. code:: python

    subgraph_filename = 'conf/expr0130_infoscan_simple.py'

    # the current graph
    graph = OrderedDict([
        ("reuser", {
            # a Block2
            'block': Block2,
            'params': {
                'topblock': False,
                'numsteps': 1,
                # points to the a config file where the subgraph is specified
                'subgraph': subgraph_filename,
                # local overrides for the subgraph configuration
                'subgraphconf': {
                    # the format is: 'block-id/block-param-key': block-param-value
                    'puppylog/file': {'filename': cnf['logfile']},
                    'puppylog/type': cnf['logtype'],
                    'puppylog/blocksize': numsteps,
                }
            },
        }),
    ])

.. note:: This is not fully tested, especially not for nested graphs and mixing with loopblocks.

2-LoopBlock2
++++++++++++

Using a loopblock with one loop item is equivalent to the single
subgraph inclusion above. The Loopblock's parameters are modified via
the loop mechanism.

FIXME: Config example

3-Block2 / lconf
++++++++++++++++

The special variable 'lconf' (local conf) can be used. During config
load any included config file's lconf dict is replaced with the
current configuration's lconf dict. A preliminary convention is a
correspondence of lconf variable- and global variable names. The lconf
dict is passed into Block2 via an `'lconf': {...}` param.

FIXME: Config example
"""
__all__ = [
    'block_cls', 'block_cls_ros',
    'block_meas_infth', 'block_meas', 'block_models', 'block_ols',
    'block_plot', 'block', 'common', 'experiment',
    'funcs', 'graph', 'utils_conf', 'utils_logging', 'utils'
]

# FIXME: deal with essentia on/off switch here?
# 'block_meas_essentia',    

from smp_graphs import *
