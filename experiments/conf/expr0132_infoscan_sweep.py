"""Reuse and modify expr0130
"""
graphconf = 'conf/expr0130_infoscan_simple.py'
debug = False

# add latex output
outputs = {
    'latex': {'type': 'latex',},
}

# local configuration
cnf = {
    'ydim_eff': 1,
    'logfile': 'data/sin_sweep_0-6.4Hz_newB_p3.pkl',
    'numsteps': 1000, # 5000
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
}

expr_number = 19
expr_name = 'Experiment {0}'.format(expr_number)
numsteps = cnf['numsteps']
lconf = {
    'delay_embed_len': 1,
}

desc = "Repeating expr0130 with a frequency swept sinusoid motor signal."

# graph
graph = OrderedDict([
    ("reuser", {
        'block': Block2,
        'params': {
            # 'debug': True,
            'topblock': False,
            'numsteps': 1,
            'lconf': lconf,
            # points to config file containing the subgraph specification
            'subgraph': graphconf,
            # dynamic id rewriting on instantiation
            'subgraph_rewrite_id': False,
            # override initial config with local config
            'subgraphconf': {
                'puppylog/file': {'filename': cnf['logfile']},
                'puppylog/type': cnf['logtype'],
                'puppylog/blocksize': numsteps,
            }
        },
    }),
])
