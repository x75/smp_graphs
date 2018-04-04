"""Reuse and modify dm_imol
"""
graphconf = 'conf/dm_imol.py'
debug = False

# add latex output
outputs = {
    'latex': {'type': 'latex',},
}

# local configuration
cnf = {
    'numsteps': 2000, # 5000
}
    
numsteps = cnf['numsteps']
lconf = {
}

desc = """Variation 1 of dm_imol using a different target function.""".format()

# graph
graph = OrderedDict([
    ("reuser", {
        'block': Block2,
        'params': {
            'debug': True,
            'topblock': False,
            'numsteps': 1,
            'lconf': lconf,
            # points to config file containing the subgraph specification
            'subgraph': graphconf,
            # dynamic id rewriting on instantiation
            'subgraph_rewrite_id': False,
            # override initial config with local config
            'subgraphconf': {
                # 'puppylog/file': {'filename': cnf['logfile']},
                # 'puppylog/type': cnf['logtype'],
                # 'puppylog/blocksize': numsteps,
            }
        },
    }),
])

