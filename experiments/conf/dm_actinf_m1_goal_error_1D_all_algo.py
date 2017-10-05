"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reuse
# execution
saveplot = True
recurrent = True
debug = False
showplot = True

# experiment
randseed = 12345
numsteps = int(10000/5)
dim = 1
motors = dim
dt = 0.1

# looparray = [('file', {'filename': fname, 'filetype': 'puppy', 'offset': random.randint(0, 500), 'length': numsteps}) for fname in filearray]
models = ['knn', 'igmm', 'soesgp'] # , 'storkgp',]
looparray = [('subgraphconf', {
    'pre_l0/models': {
        'm1': {
            'type': 'actinf_m1',
            'algo': model,
            # 'type': 'actinf_m1',
            # 'algo': algo,
            # 'lag_past': lag_past,
            # 'lag_future': lag_future,
            # 'idim': dim_s_proprio * (lag_past[1] - lag_past[0]) * 2, # laglen
            # 'odim': dim_s_proprio * (lag_future[1] - lag_future[0]), # laglen,
            # 'laglen': laglen,
            # 'eta': eta,
        },
    },
    'plot_ts/title': 'actinf_m1 1D with %s' % (model, )
    }) for (i, model) in enumerate(models)]

# 'pre_l0_test/models': {'fwd': {'type': 'actinf_m1', 'algo': 'copy', 'copyid': 'pre_l0', 'idim': dim * 2, 'odim': dim},},

loopblock = {
    'block': Block2,
    'params': {
        'id': 'actinf_m1_loop',
        'debug': False,
        'blocksize': 1,
        'subgraph': 'conf/dm_actinf_m1_goal_error_ND_embedding.py',
        'subgraphconf': {},
        'subgraph_rewrite_id': True,
        'outputs': {},
        },
    }

# graph
graph = OrderedDict([
    # puppy data
    ('actinf_m1_loop', {
        'block': LoopBlock2,
        'blocksize': 1,
        'params': {
            'id': 'actinf_m1_loop',
            'loop': looparray,
            # required
            'loopmode': 'parallel',
            'loopblock': loopblock,
        },
    }),
])
