"""smp_graphs testing hierarchical composition config
"""

# reused variables
graphconf = 'conf/default2_loop.py'; numsteps = 1000
# graphconf = 'conf/default2.py' ; numsteps = 100
debug = False
# graph
graph = OrderedDict([
    # a constant
    ("bhier", {
        'block': Block2,
        'params': {
            'id': 'bhier',
            'debug': False,
            'topblock': False,
            'numsteps': 1,
            # contains the subgraph specified in this config file
            'subgraph': graphconf,
        },
    }),
])
