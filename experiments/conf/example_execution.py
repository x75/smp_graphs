"""smp_graphs config

test execution timing issues:
 - blocksize
 - rate
 - input shape / ibuf
 - output shape / obuf
 - input shape vs. bus/shape"""

randseed = 0

numsteps = 100

# inputs
# inkey = str: inval = {val: value, shape: shape, src: source, ...}

# outputs
# outkey = str: outval = {val: value, shape: shape, dst: destination, ...}

# bus
# buskey = str: busval = {val: value, shape: shape, src: source, dst: [destinations]}

graph = OrderedDict([
    ('cnt1', {
        'block': CountBlock2,
        'params': {
            'blocksize': 8,
            'debug': False,
            'inputs': {},
            'outputs': {'x': {'shape': (5, 2, 3, 8)}},
        },
    }),
    
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
            'outputs': {'x': {'shape': (5, 2, 3, 1)}},
            # 'outputs': {'x': {'shape': (2, 3,)}},
            # 'outputs': {'x': {'shape': (3,)}},
        },
    }),
    
    ('src2', {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'src2',
            'blocksize': 10,
            'debug': False,
            'rate': 20,
            # recurrent connection
            'inputs': {'lo': {'bus': 'src2/x'}, 'hi': {'bus': 'src1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
            'outputs': {'x': {'shape': (5, 2, 3, 10)}},
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
            'ylim_share': False,
            'inputs': {'d0': {'bus': 'cnt1/x', 'shape': (5, 2, 3, numsteps)},
                       'd1': {'bus': 'src1/x', 'shape': (5, 2, 3, numsteps)},
                       'd2': {'bus': 'src2/x', 'shape': (5, 2, 3, numsteps)}},
            'subplots': [
                [
                    {'input': ['d0'], 'shape': (3, 50), 'plot': timeseries, 'xslice': (0, 50), 'ndslice': (slice(50), 0,0,None)},
                    {'input': ['d1'], 'shape': (3, 80), 'plot': timeseries, 'xslice': (0, 80), 'ndslice': (slice(80), 0, None, 0)},
                    {'input': ['d2'], 'shape': (3, numsteps), 'ndslice': (slice(None), 0, 0, None), 'plot': timeseries},
                ],
            ]
        },
    })
])
