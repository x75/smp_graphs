"""smp_graphs config

test multi-dimensional bus"""

randseed = 0

numsteps = 20

# inputs
# inkey = str: inval = {val: value, shape: shape, src: source, ...}

# outputs
# outkey = str: outval = {val: value, shape: shape, dst: destination, ...}

# bus
# buskey = str: busval = {val: value, shape: shape, src: source, dst: [destinations]}

graph = OrderedDict([
    ('src1', {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'src1',
            'blocksize': 1,
            'debug': False,
            'inputs': {
                'lo': {'val': 0, 'shape': (2, 3, 1), 'src': 'const'},
                'hi': {'val': 1, 'shape': (2, 3, 1), 'src': 'const'}
            },
            # recurrent connection
            # 'inputs': {'lo': {'bus': 'b2/x'}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'outputs': {'x': {'shape': (5, 2, 3,)}},
            # 'outputs': {'x': {'shape': (2, 3,)}},
            # 'outputs': {'x': {'shape': (3,)}},
        },
    }),
    
    ('src2', {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'src2',
            'blocksize': 1,
            'debug': False,
            'rate': 2,
            # recurrent connection
            'inputs': {'lo': {'bus': 'src2/x'}, 'hi': {'bus': 'src1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
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
            'debug': False,
            'inputs': {'d1': {'bus': 'src1/x'}, 'd2': {'bus': 'src2/x'}},
            'subplots': [
                [
                    {'input': ['d1'], 'shape': (2, 3, numsteps), 'plot': timeseries},
                    {'input': ['d2'], 'shape': (2, 3, numsteps), 'plot': timeseries},
                ],
            ]
        },
    })
])
