"""Reuse and modify expr0161
"""
graphconf = 'conf/expr0161_infoscan_elementwise.py'
debug = False

# add latex output
outputs = {
    'latex': {'type': 'latex',},
}

# local configuration
cnf = {
    'ydim_eff': 1,
    'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle',
    'numsteps': 1000, # 5000
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
}

desc = """Repeating the elementwise scan of expr0161 in
\\ref{{sec:smp-expr0161-infoscan-elementwise}} with a the sweep
exploration signal already used in expr0132\\_infoscan\\_sweep and
expr0150\\_infoscan\\_windowed brings out additional details about the
motor-sensor couplings of the Puppy robot. First, the low-frequency
resonances are not produced, making the mutual information agree more
with the conditional measures. The sweep signal's actual transfer of
information is larger than for the high bandwidth square pulses of the
periodic exploration signal. Also, the information is transferred as a
compact packet instead of the intermittent response to the square
pulses. This is again a clear demonstration that an agent's
action-delay expectation, or by extension its body schema, is a
dynamic entity with a potentially important role for both
introspective (self-state) as well as predictive functions."""

numsteps = cnf['numsteps']
lconf = {
    'delay_embed_len': 1,
}

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
                'puppylog/file': {'filename': cnf['logfile']},
                'puppylog/type': cnf['logtype'],
                'puppylog/blocksize': numsteps,
            }
        },
    }),
])
