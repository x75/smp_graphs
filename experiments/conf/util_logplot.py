"""smp_graphs default conf

the config is python, so we
 - import stuff we need in the config
 - put the graph config into a dict
"""

# reuse
numsteps = 10
debug = True

# graph
graph = OrderedDict([
    # a constant
    ("selflog", {
        'block': FileBlock2,
        'params': {
            'id': 'selflog',
            'logging': False,
            'inputs': {},
            'debug': False,
            'blocksize': numsteps,
            'type': 'selflog',
            # this is looping demand
            'file': {'filename':
                'data/experiment_20170519_123619_default2_multi_dim_bus_pd.h5',
                         # 'data/experiment_20170510_124440_puppy_rp_pd.h5',
                # 'data/experiment_20170510_123450_puppy_rp_pd.h5',
                # 'data/experiment_20170510_121248_puppy_rp_pd.h5',
                # 'data/experiment_20170509_131125_puppy_rp_blocksize_pd.h5',
                # 'data/experiment_20170507_154742_pd.h5',
                # 'data/experiment_20170505_111138_pd.h5', # puppy_rp, 500 steps
                # 'data/experiment_20170505_110833_pd.h5' # default2_loop 1000 steps
                # 'data/experiment_20170505_084006_pd.h5'
                # 'data/experiment_20170505_083801_pd.h5',
                # 'data/experiment_20170505_003754_pd.h5',
                # 'data/experiment_20170505_001511_pd.h5',
                # 'data/experiment_20170505_001143_pd.h5',
                # 'data/experiment_20170505_000540_pd.h5',
                # 'data/experiment_20170504_192821_pd.h5',
                # 'data/experiment_20170504_202016_pd.h5',
                # 'data/experiment_20170504_222828_pd.h5',
                },
            # 'outputs': {'conf': [(1,1)], 'conf_final': [(1,1)]},
            'outputs': {'log': [None], 'x': {'shape': (3, numsteps)}},
        },
    }),
    ('plotter', {
        'block': PlotBlock2,
        'params': {
            'id': 'plotter',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                # 'd1': ['selflog//b1/x'],
                # 'd2': ['selflog//b2/x'],
                'd3': {'bus': 'selflog/x', 'shape': (3, numsteps)},
                # 'd4': ['selflog/y'],
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                # [
                #     {'input': 'd1', 'slice': (0, 3), 'plot': timeseries},
                #     # {'input': 'd1', 'slice': (0, 3), 'plot': histogram},
                # ],
                # [
                #     {'input': 'd2', 'slice': (3, 6), 'plot': timeseries},
                #     # {'input': 'd2', 'slice': (3, 6), 'plot': histogram},
                # ],
                [
                    {'input': 'd3', 'slice': (3, 6), 'plot': timeseries},
                    # {'input': 'd3', 'slice': (3, 6), 'plot': histogram},
                ],
                # [
                #     {'input': 'd4', 'slice': (3, 6), 'plot': timeseries},
                #     # {'input': 'd4', 'slice': (3, 6), 'plot': histogram},
                # ],
            ]
        },
    }),
])
