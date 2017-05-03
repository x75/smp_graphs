"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# would like to get rid of this, common for all conf

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

# from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block import Block2, ConstBlock2, UniformRandomBlock2
from smp_graphs.block_plot import TimeseriesPlotBlock2
# from smp_graphs.block_plot import TimeseriesPlotBlock

from smp_base.plot import timeseries, histogram

import numpy as np

# reuse
numsteps = 100

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
            'outputs': {'x': [(3,1)]},
            'debug': False,
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'idim': 6,
            'odim': 3,
            # 'lo': 0,
            # 'hi': 1,
            'outputs': {'x': [(3, 1)]},
            'debug': True,
            # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            # recurrent connection
            'inputs': {'lo': ['b2/x'], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
        },
    }),
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 6,
            'odim': 3,
            'debug': True,
            'inputs': {'d1': ['b1/x'], 'd2': ['b2/x']},
            'outputs': {'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        }
    })
])

# top block
conf = {
    'block': Block2,
    'params': {
        'id': make_expr_id(),
        'debug': True,
        'topblock': True,
        "numsteps": numsteps,
        "graph": graph,
    }
}
