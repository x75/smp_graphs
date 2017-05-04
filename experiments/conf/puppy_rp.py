"""smp_graphs puppy recurrence plot conf
"""

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block2, ConstBlock2, UniformRandomBlock2
from smp_graphs.block import FileBlock2

from smp_graphs.block_plot import TimeseriesPlotBlock2

from smp_base.plot import timeseries, histogram
from smp_base.plot import rp_timeseries_embedding

import numpy as np

# reuse
numsteps = 1000

# files array or enclosing loop block?

# graph
graph = OrderedDict([
    # puppy data
    ('puppydata', {
        'block': FileBlock2,
        'params': {
            'id': 'puppydata',
            'idim': None,
            'odim': 'auto',
            'debug': False,
            'blocksize': numsteps,
            'file': [
                'data/pickles_puppy_03_22_14U/recording_eC0.41_eA0.03_c0.50_n1000_id0.pickle',
            ],
            'outputs': {'x': [None], 'y': [None]},
        },
    }),
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
            'outputs': {'x': [(3, 1)]},
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
            'inputs': {'lo': [0], 'hi': ['b1/x']},
            'outputs': {'x': [(3, 1)]},
            'debug': False,
        },
    }),
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 4,
            'odim': 1,
            'debug': False,
            'inputs': {
                'd1': ['puppydata/x'],
                'd2': ['puppydata/y'],
                'd3': ['b2/x']},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 6), 'plot': timeseries},
                    {'input': 'd1', 'slice': (0, 6), 'plot': histogram},
                    {'input': 'd1', 'slice': (0, 6), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'd2', 'slice': (0, 4), 'plot': timeseries},
                    {'input': 'd2', 'slice': (0, 4), 'plot': histogram},
                    {'input': 'd2', 'slice': (0, 4), 'plot': rp_timeseries_embedding},
                ],
                [
                    {'input': 'd3', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd3', 'slice': (0, 3), 'plot': histogram},
                    {'input': 'd3', 'slice': (0, 3), 'plot': rp_timeseries_embedding},
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
        'topblock': True,
        "numsteps": numsteps,
        "graph": graph
    }
}
