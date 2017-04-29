"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# imports
from smp_graphs.experiment import make_expr_id

from collections import OrderedDict

from smp_graphs.block import Block, ConstBlock, UniformRandomBlock
from smp_graphs.block_plot import TimeseriesPlotBlock

# how can we avoid these imports, *?
from plot import timeseries, histogram

import numpy as np

# reused variables
numsteps = 100

# comp. graph
graph = OrderedDict([
    ("b1", {
        'block': ConstBlock,
        'params': {
            'id': 'b1',
            'idim': None,
            'odim': 3,
            'const': np.random.uniform(-1, 1, (3, 1)),
        },
    }),
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
    ("bplot", {
        'block': TimeseriesPlotBlock,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'idim': 6,
            'odim': 3,
            'debug': False,
            'inputs': ['b1', 'b2'],
            'subplots': [
                [
                    {'inputs': (0, 3), 'plot': timeseries},
                    {'inputs': (0, 3), 'plot': histogram},
                ],
                [
                    {'inputs': (3, 6), 'plot': timeseries},
                    {'inputs': (3, 6), 'plot': histogram},
                ],
            ]
        }
    })
])

# top block config, sorry
conf = {
    'block': Block,
    'params': {
        'id': make_expr_id(),
        'topblock': True,
        "numsteps": numsteps,
        "graph": graph
    }
}
