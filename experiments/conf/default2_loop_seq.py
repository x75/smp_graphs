"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

from smp_graphs.block import CountBlock2
from smp_graphs.block import SeqLoopBlock2

from functools import partial

# reused variables
numsteps = 100

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': CountBlock2,
        'params': {
            'id': 'b1',
            'inputs': {},
            'outputs': {'cnt_': [(1,1)]},
            'blocksize': 10,
            'debug': False,
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': SeqLoopBlock2,
        'params': {
            'id': 'b2',
            'loop': [('inputs', {'c': [np.random.uniform(-i, i, (3, 1))]}) for i in range(1, 4)],
            'loopmode': 'parallel',
            'loopblock': {
                'block': ConstBlock2,
                'params': {
                    'id': 'b3',
                    'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
                    'outputs': {'x': [(3,1)]},
                    'debug': False,
                },
            },
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
            'inputs': {'d1': ['b1/cnt_']}, # 'd2': ['b2/x']},
            'outputs': {'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 3), 'plot': partial(timeseries, marker = 'o', linestyle = 'None')},
                    {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                ],
                # [
                #     {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                #     {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
            ]
        }
    })
])
