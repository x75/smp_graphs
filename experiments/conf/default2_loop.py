"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reuse
numsteps = 1000

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            # 'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
            # 'outputs': {'x': [(3,1)]},
            'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 1)), 'shape': (3,)}},
            'outputs': {'x': {'shape': (3,)}},
            'debug': False,
            'blocksize': 1,
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
            'outputs': {'x': {'shape': (3, )}},
            'debug': False,
            'inputs': {'lo': {'val': 0, 'shape': (3,)}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'blocksize': 1,
        },
    }),
    # loop block test
    ('b3', {
        'block': LoopBlock2,
        'params': {
            'id': 'b3',
            'loop': [('inputs', {'c': {'val': np.random.uniform(-i, i, (3, 1)), 'shape': (3,)}}) for i in range(1, 4)],
            # 'loopmode': 'parallel',
            'loopblock': {
                'block': ConstBlock2,
                'params': {
                    'id': 'b3',
                    'inputs': {'c': {'val': np.random.uniform(-1, 1, (3, 1))}},
                    'outputs': {'x': {'shape': (3,)}},
                    'debug': True,
                },
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
            'idim': 6,
            'odim': 3,
            'debug': False,
            'inputs': {'d1': {'bus': 'b1/x'}, 'd2': {'bus': 'b2/x'}, 'd3': {'bus': 'b3_1/x'}, 'd4': {'bus': 'b3_2/x'}, 'd5': {'bus': 'b3_3/x'}},
            'outputs': {'x': {'shape': (3, )}},
            'subplots': [
                [
                    {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                    {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                ],
                [
                    {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                    {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                ],
            ] + [[
                {'input': 'd%d' % i, 'slice': (1, 1), 'plot': timeseries},
                {'input': 'd%d' % i, 'slice': (1, 1), 'plot': histogram},
                ] for i in range(3, 6)]
        }
    })
])
