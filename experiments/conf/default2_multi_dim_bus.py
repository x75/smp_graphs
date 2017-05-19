"""smp_graphs config

test multi-dimensional bus"""

randseed = 0

numsteps = 10

# inputs
# inkey = str: inval = {val: value, shape: shape, src: source, ...}

# outputs
# outkey = str: outval = {val: value, shape: shape, dst: destination, ...}

# bus
# buskey = str: busval = {val: value, shape: shape, src: source, dst: [destinations]}

graph = OrderedDict([
    ('src', {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'src',
            'blocksize': 1,
            'debug': True,
            'inputs': {
                'lo': {'val': 0, 'shape': (2, 3, 1), 'src': 'const'},
                'hi': {'val': 1, 'shape': (2, 3, 1), 'src': 'const'}
            },
            # recurrent connection
            # 'inputs': {'lo': ['b2/x'], 'hi': ['b1/x']}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'outputs': {'x': {'shape': (5, 2, 3,)}},
            # 'outputs': {'x': {'shape': (2, 3,)}},
            # 'outputs': {'x': {'shape': (3,)}},
        },
    }),
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'blocksize': numsteps,
            'debug': True,
            'inputs': {'d1': {'bus': 'src/x'}},
            'subplots': [
                [
                    {'input': ['d1'], 'shape': (2, 3, numsteps), 'plot': timeseries}
                ],
            ]
        },
    })
])
