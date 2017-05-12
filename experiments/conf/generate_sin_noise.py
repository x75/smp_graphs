"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

from smp_graphs.funcs import f_sin, f_sin_noise
from smp_graphs.block_plot import SnsMatrixPlotBlock2
from smp_base.plot import histogramnd

# reused variables
numsteps = 1000

# graph
graph = OrderedDict([
    # a constant
    ("cnt", {
        'block': CountBlock2,
        'params': {
            'id': 'cnt',
            'inputs': {},
            'outputs': {'x': [(1,1)]},
            'debug': False,
        },
    }),
    # a random number generator, mapping const input to hi
    ("sin", {
        'block': FuncBlock2,
        'params': {
            'id': 'sin',
            'outputs': {'y': [(1, 1)]},
            'debug': False,
            # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            # recurrent connection
            'inputs': {'x': ['cnt/x'],
                       'f': [np.array([[0.003]])],
                       'sigma': [0.3]}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'func': f_sin_noise,
        },
    }),
    
    # a random number generator, mapping const input to hi
    ("noise", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'noise',
            'outputs': {'r': [(1, 1)]},
            'debug': False,
            # 'inputs': {'lo': [0, (3, 1)], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            # recurrent connection
            'inputs': {'lo': [np.array([[-0.1]])],
                       'hi': [np.array([[ 0.1]])],
            },
        },
    }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("bplot", {
        'block': PlotBlock2,
        'params': {
            'id': 'bplot',
            'blocksize': numsteps,
            'debug': False,
            'inputs': {'d1': ['cnt/x'], 'd2': ['sin/y'], 'd3': ['noise/r']},
            'outputs': {},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                ],
                [
                    {'input': 'd3', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd3', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        }
    }),
    ('plot2', {
        'block': SnsMatrixPlotBlock2,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                'd1': ['cnt/x'],
                'd2': ['sin/y'],
                'd3': ['noise/r'],
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                [
                    # stack inputs into one vector (stack, combine, concat
                    {'input': ['d1', 'd2', 'd3'], 'mode': 'stack', 'plot': histogramnd},
                ],
            ],
        },
    }),
])
