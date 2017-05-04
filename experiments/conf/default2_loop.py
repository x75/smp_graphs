"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reuse
numsteps = 10

# graph
graph = OrderedDict([
    # a constant
    ("b1", {
        'block': ConstBlock2,
        'params': {
            'id': 'b1',
            'inputs': {'c': [np.random.uniform(-1, 1, (3, 1))]},
            'outputs': {'x': [(3,1)]},
            'debug': False,
        },
    }),
])
