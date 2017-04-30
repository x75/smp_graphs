"""smp_graphs puppy recurrence plot conf
"""

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block import FileBlock

from smp_graphs.block_plot import TimeseriesPlotBlock

from smp_base.plot import timeseries, histogram

import numpy as np

# reuse
numsteps = 100

# files array or enclosing loop block?

# graph
graph = OrderedDict([
    # puppy data
    ('puppydata', {
        'block': FileBlock,
        'params': {
            'id': 'puppydata',
            'idim': None,
            'odim': 1, # 'auto',
            'debug': True,
            'file': [
                'data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle',
            ]
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
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock,
        'params': {
            'id': 'b2',
            'idim': 6,
            'odim': 3,
            'lo': 0,
            'hi': 1,
            'inputs': ['b1']
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
            'odim': 3,
            'debug': True,
            'inputs': ['puppydata', 'b2'],
            'subplots': [
                [
                    {'inputs': (0, 1), 'plot': timeseries},
                    {'inputs': (0, 1), 'plot': histogram},
                ],
                [
                    {'inputs': (1, 4), 'plot': timeseries},
                    {'inputs': (1, 4), 'plot': histogram},
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
