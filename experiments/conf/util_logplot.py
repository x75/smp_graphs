"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""


selflogcnf = {
    'numsteps': 10,
    'filetype': 'selflog',
    'filename': 'data/experiment_20170519_123619_default2_multi_dim_bus_pd.h5',
    'xdim': 3,
}

puppypklcnf = {
    'numsteps': 5000,
    'filetype': 'puppy',
    'filename': 'data/sin_sweep_0-6.4Hz_newB.pickle',
    'xdim': 6,
    'ydim': 4,
}

cnf = puppypklcnf
    
# reuse
numsteps = cnf['numsteps']
debug = True

# graph
graph = OrderedDict([
    # a constant
    ("selflog", {
        'block': FileBlock2,
        'params': {
            'id': 'selflog',
            'logging': False,
            'inputs': {},
            'debug': False,
            'blocksize': numsteps,
            'type': cnf['filetype'],
            # this is looping demand
            'file': {'filename':
                cnf['filename'],
            },
            # 'outputs': {'conf': [(1,1)], 'conf_final': [(1,1)]},
            'outputs': {'log': [None], 'x': {'shape': (cnf['xdim'], numsteps)}, 'y': {'shape': (cnf['ydim'], numsteps)}},
        },
    }),
    ('plotter', {
        'block': PlotBlock2,
        'params': {
            'id': 'plotter',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                # 'd1': ['selflog//b1/x'],
                # 'd2': ['selflog//b2/x'],
                'd3': {'bus': 'selflog/x', 'shape': (cnf['xdim'], numsteps)},
                # 'd4': ['selflog/y'],
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                # [
                #     {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                #     # {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                # ],
                # [
                #     {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                #     # {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
                [
                    {'input': 'd3', 'slice': (3, 6), 'plot': timeseries},
                    # {'input': 'd3', 'slice': (3, 6), 'plot': histogram},
                ],
                # [
                #     {'input': 'd4', 'slice': (3, 6), 'plot': timeseries},
                #     # {'input': 'd4', 'slice': (3, 6), 'plot': histogram},
                # ],
            ]
        },
    }),
])
