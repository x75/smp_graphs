"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# from smp_graphs.block import FuncBlock2
# from smp_graphs.block_models import ModelBlock2

randseed = 24
# reused variables
numsteps = 8000

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'blocksize': 1,
            'inputs': {'c': {'val': np.random.uniform(-1, 0.999, (3, 1))}},
            'outputs': {'x': {'shape': (3,1)}},
            'debug': False,
        },
    }),
    
    # a random number generator, mapping const input to hi
    ("b2", {
        'block': UniformRandomBlock2,
        'params': {
            'id': 'b2',
            'blocksize': 1,
            'outputs': {'x': {'shape': (3, 1)}},
            'debug': False,
            'inputs': {'lo': {'val': np.zeros((3, 1))}, 'hi': {'bus': 'b1/x'}}, # , 'li': np.random.uniform(0, 1, (3,)), 'bu': {'b1/x': [0, 1]}}
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
            'debug': True,
            'saveplot': False,
            'savetype': 'pdf',
            'wspace': 0.2, 'hspace': 0.2,
            'inputs': {'d1': {'bus': 'b1/x', 'shape': (3, numsteps)}, 'd2': {'bus': 'b2/x', 'shape': (3, numsteps)}},
            'outputs': {}, # 'x': {'shape': (3, 1)}
            'subplots': [
                [
                    {'input': 'd1', 'ndslice': (slice(None), slice(None)), 'shape': (3, numsteps), 'plot': 'timeseries'},
                    {'input': 'd1', 'ndslice': (slice(None), slice(None)), 'shape': (3, numsteps),
                         'plot': 'partial(histogram, orientation = \'horizontal\')'},
                ],
                [
                    # {'input': 'd2', 'ndslice': (slice(None), slice(2)), 'shape': (2, numsteps), 'plot': 'timeseries'},
                    {'input': 'd2', 'ndslice': (slice(None), slice(None)), 'shape': (3, numsteps), 'plot': 'timeseries'},
                    {'input': 'd2', 'ndslice': (slice(None), slice(None)), 'shape': (3, numsteps),
                         'plot': 'partial(histogram, orientation = \'horizontal\')'},
                ],
            ]
        }
    })
])
