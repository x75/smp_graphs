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
    ("bhier", {
        'block': Block2,
        'params': {
            'id': 'b1',
            'debug': True,
            'topblock': False,
            'numsteps': numsteps,
            'subblock': 'conf/default2.py',
        },
    }),
])
