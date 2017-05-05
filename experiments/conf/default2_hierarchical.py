"""smp_graphs testing hierarchical composition config
"""

# reused variables
graphconf = 'conf/default2_loop.py'; numsteps = 1000
# graphconf = 'conf/default2.py' ; numsteps = 100

# graph
graph = OrderedDict([
    # a constant
    ("bhier", {
        'block': Block2,
        'params': {
            'id': 'b1',
            'debug': False,
            'topblock': False,
            'numsteps': numsteps,
            # contains the subgraph specified in this config file
            'subgraph': graphconf,
        },
    }),
])
