"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reused variables
numsteps = 100

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 2, 1))}},
            'outputs': {'x': {'shape': (3,2)}},
            'debug': True,
        },
    }),
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'outputs': {'x': {'shape': (3, 2)}},
            'debug': False,
            'inputs': {'lo': {'val': np.zeros((3, 2, 1))}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            # recurrent connection
            # 'inputs': {'lo': {'bus': 'b2/x'}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
        },
    }),
    
    # plot module with blocksize = episode, fetching input from busses
    # and mapping that onto plots
    ("plot", {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'blocksize': numsteps,
            'debug': False,
            'saveplot': False,
            'savetype': 'pdf',
            'inputs': {'d1': {'bus': 'b1/x'}, 'd2': {'bus': 'b2/x'}},
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'plot': timeseries},
                    {'input': 'd1', 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        }
    })
])
