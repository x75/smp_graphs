"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# would like to get rid of this, common for all conf

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block_plot import TimeseriesPlotBlock

from smp_base.plot import timeseries, histogram

import numpy as np

# reuse
numsteps = 100

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock,
        'params': {
            'id': 'b1',
            'idim': None,
            'odim': 3,
            'const': np.random.uniform(-1, 1, (3, 1)),
            'outputs': {'x': [(3,1)]},
            'debug': False,
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock,
        'params': {
            'id': 'b2',
            'idim': 6,
            'odim': 3,
            # 'lo': 0,
            # 'hi': 1,
            'outputs': {'x': [(3, 1)]},
            'debug': True,
            'inputs': {'lo': 0, 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
        },
    }),
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': TimeseriesPlotBlock,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 6,
            'odim': 3,
            'debug': True,
            'inputs': {'data': ['b1/x', 'b2/x']},
            'outputs': {'x': [3]},
            'subplots': [
                [
                    {'input': 'data', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'data', 'slice': (0, 3), 'plot': histogram},
                ],
                [
                    {'input': 'data', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'data', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        }
    })
])

# top block
conf = {
    'block': Block,
    'params': {
        'id': make_expr_id(),
        'topblock': True,
        "numsteps": numsteps,
        "graph": graph,
    }
}
