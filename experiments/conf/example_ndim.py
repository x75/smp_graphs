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
            'blocksize': 10,
            'inputs': {'c': {'val': np.repeat(np.random.uniform(-1, 1, (3, 2)), 10).reshape((3,2,10)), 'shape': (3,2,10)}},
            'outputs': {'x': {'shape': (3,2,10)}},
            'debug': False,
        },
    }),
    
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'blocksize': 10,
            'outputs': {'x': {'shape': (3, 2, 10)}},
            'debug': False,
            'inputs': {'lo': {'val': np.zeros((3, 2, 10))}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
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
            'wspace': 0.2, 'hspace': 0.2,
            'inputs': {'d1': {'bus': 'b1/x', 'shape': (3, 2, numsteps)}, 'd2': {'bus': 'b2/x', 'shape': (3, 2, numsteps)}},
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'ndslice': (slice(None), slice(None), slice(None)), 'shape': (6, numsteps), 'plot': timeseries},
                    {'input': 'd1', 'ndslice': (slice(None), slice(None), slice(None)), 'shape': (6, numsteps), 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'ndslice': (slice(None), slice(None), slice(None)), 'shape': (6, numsteps), 'plot': timeseries},
                    {'input': 'd2', 'ndslice': (slice(None), slice(None), slice(None)), 'shape': (6, numsteps), 'plot': histogram},
                ],
            ]
        }
    })
])
