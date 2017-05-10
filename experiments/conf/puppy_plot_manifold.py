"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

numsteps = 147000

graph = OrderedDict([
    ('puppylog', {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'inputs': {},
            'type': 'selflog',
            'file': [
                'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5',
                ],
            'blocksize': numsteps,
            'outputs': {'log': [None]},
            }
        }),
    ('plot', {
        'block': TimeseriesPlotBlock2,
        'params': {
            'id': 'plot',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                'd3': ['puppylog/x'],
                'd4': ['puppylog/y']
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
                [
                    {'input': 'd4', 'slice': (3, 6), 'plot': timeseries},
                    # {'input': 'd4', 'slice': (3, 6), 'plot': histogram},
                ],
            ]
        },
    })
])
