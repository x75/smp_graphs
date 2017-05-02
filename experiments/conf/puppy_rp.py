"""smp_graphs puppy recurrence plot conf
"""

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block import FileBlock

from smp_graphs.block_plot import TimeseriesPlotBlock

from smp_base.plot import timeseries, histogram
from smp_base.plot import rp_timeseries_embedding

import numpy as np

# reuse
numsteps = 20

# files array or enclosing loop block?

# graph
graph = OrderedDict([
    # puppy data
    ('puppydata', {
        'block': FileBlock,
        'params': {
            'id': 'puppydata',
            'idim': None,
            'odim': 'auto',
            'debug': True,
            'blocksize': numsteps,
            'file': [
                'data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle',
            ],
            'outputs': {'x': [0], 'y': [0]},
        },
    }),
    # a constant
    ("b1", {
        'block': ConstBlock,
        'params': {
            'id': 'b1',
            'idim': None,
            'odim': 3,
            'const': np.random.uniform(-1, 1, (3, 1)),
            'outputs': {'x': [3]},
            'debug': True,
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
            'inputs': {'lo': 0, 'hi': ['b1/x']},
            'outputs': {'x': [3]},
            'debug': True,
        },
    }),
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': TimeseriesPlotBlock,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 4,
            'odim': 1,
            'debug': True,
            'inputs': {
                'puppydata/x': ['puppydata/x'],
                'puppydata/y': ['puppydata/y'],
                'b2/x': ['b2/x']},
            'subplots': [
                [
                    {'input': 'puppydata/x', 'slice': (0, 6), 'plot': timeseries},
                    {'input': 'puppydata/x', 'slice': (0, 6), 'plot': histogram},
                    {'input': 'puppydata/x', 'slice': (0, 6), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'puppydata/y', 'slice': (0, 4), 'plot': timeseries},
                    {'input': 'puppydata/y', 'slice': (0, 4), 'plot': histogram},
                    {'input': 'puppydata/y', 'slice': (0, 4), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'b2/x', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'b2/x', 'slice': (0, 3), 'plot': histogram},
                    {'input': 'b2/x', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
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
        "graph": graph
    }
}
