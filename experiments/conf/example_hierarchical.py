"""smp_graphs testing hierarchical composition config
"""

# reused variables
# the outer numsteps need to match the inner numsteps (FIXME), otherwise broadcasting takes place
graphconf = 'conf/example_loop.py'; numsteps = 100
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
