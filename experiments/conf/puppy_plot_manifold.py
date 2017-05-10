"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd

numsteps = 147000
# numsteps = 2000

graph = OrderedDict([
    ('puppylog', {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'inputs': {},
            'type': 'selflog',
            'file': [
                # all files
                'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5',
                # short version
                # 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5',
                ],
            'blocksize': numsteps,
            'outputs': {'log': [None]},
            }
        }),
    # puppy process data block: integrate acc, diff motors
    # ('puppyproc', {}),
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
                    {'input': 'd4', 'slice': (3, 6), 'plot': timeseries},
                ],
                [
                    # stack inputs into one vector (stack, combine, concat
                    {'input': ['d3', 'd4'], 'mode': 'stack',
                         'slice': (3, 6), 'plot': histogramnd},
                    {'input': [], 'plot': timeseries},
                    
                ],
            ]
        },
    })
])
