"""smp_graphs configuration

self-exploration (se)

plot different target functions
"""

ros = False
numsteps = 1000

# graph
graph = OrderedDict([
    ('cnt', {
        'block': CountBlock2,
        # 'params': {
        #     'blocksize': 1,
        #     'debug': False,
        #     'inputs': {},
        #     'outputs': {'x': {'shape': (dim_s_proprio, 1)}},
        #     },
        }),
    ('f', {
        'block': FuncBlock2,
        # 'params': {'func': lambda x: x, },
        }),
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'inputs': {
                'cnt': {'bus': 'cnt/x', 'shape': (1, numsteps)},
                'x': {'bus': 'f/x', 'shape': (1, numsteps)},
                },
            'subplots': [
                [
                    {'input': ['cnt', 'x']},
                    ]
                ]
            },
        }),
])
