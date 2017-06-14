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
numsteps = 1000
dim = 1
motors = dim
dt = 0.1

# looparray = [('file', {'filename': fname, 'filetype': 'puppy', 'offset': random.randint(0, 500), 'length': numsteps}) for fname in filearray]
models = ['knn', 'soesgp']#, 'storkgp']
looparray = [('subgraphconf', {
    'pre_l0/models': {'fwd': {'type': 'actinf_m1', 'algo': model, 'idim': dim * 2, 'odim': dim},},
    }) for (i, model) in enumerate(models)]

loopblock = {
    'block': Block2,
    'params': {
        'id': 'actinf_m1_loop',
        'debug': False,
        'blocksize': 1,
        'subgraph': 'conf/actinf_m1_goal_error_pm1d.py',
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
            'loopblock': loopblock,
        },
    }),

    # # plot module with blocksize = episode, fetching input from busses
    # # and mapping that onto plots
    # ("bplot", {
    #     'block': PlotBlock2,
    #     'params': {
    #         'id': 'bplot',
    #         'blocksize': numsteps,
    #         'idim': 6,
    #         'odim': 3,
    #         'debug': False,
    #         'inputs': {'d1': {'bus': 'b1/x', 'shape': (3, numsteps)},
    #                    'd2': {'bus': 'b2/x', 'shape': (3, numsteps)},
    #                    'd3': {'bus': 'b3_0/x', 'shape': (3, numsteps)},
    #                    'd4': {'bus': 'b3_1/x', 'shape': (3, numsteps)},
    #                    'd5': {'bus': 'b3_2/x', 'shape': (3, numsteps)}},
    #         'outputs': {},
    #         'subplots': [
    #             [
    #                 {'input': 'd1', 'xslice': (0, numsteps), 'plot': timeseries},
    #                 {'input': 'd1', 'xslice': (0, numsteps), 'plot': histogram},
    #             ],
    #             [
    #                 {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
    #                 {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
    #             ],
    #         ] + [[
    #             {'input': 'd%d' % i, 'slice': (1, 1), 'plot': timeseries},
    #             {'input': 'd%d' % i, 'slice': (1, 1), 'plot': histogram},
    #             ] for i in range(3, 6)
    #             ]
    #     }
    # })
])
