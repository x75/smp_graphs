"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2

# numsteps = 147000
# numsteps = 10000
numsteps = 2000

graph = OrderedDict([
    ('puppylog', {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'inputs': {},
            'type': 'selflog',
            'file': [
                # all files 147000
                # 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5',
                # medium version 10000
                # 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5',
                # short version 2000
                'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5',
                ],
            'blocksize': numsteps,
            'outputs': {'log': [None]},
            }
        }),
    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslice',
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'x': ['puppylog/x']},
            'slices': {'x': {'acc': slice(0, 3), 'gyr': slice(3, 6)}},
            }
        }),    
    # puppy process data block: integrate acc, diff motors
    ('accint', {
        'block': IBlock2,
        'params': {
            'id': 'accint',
            'blocksize': numsteps,
            'inputs': {'x_acc': ['puppyslice/x_acc']},
            'outputs': {},
            'd': 0.1,
            'leak': 0.01,
            },
        }),
    # puppy process data block: integrate acc, diff motors
    ('motordiff', {
        'block': dBlock2,
        'params': {
            'id': 'motordiff',
            'blocksize': numsteps,
            'inputs': {'y': ['puppylog/y']},
            'outputs': {},
            'd': 0.1,
            'leak': 0.01,
            },
        }),
    # puppy process data block: delay motors by lag to align with their sensory effects
    ('motordel', {
        'block': DelayBlock2,
        'params': {
            'id': 'motordel',
            'blocksize': numsteps,
            # 'inputs': {'y': ['motordiff/dy']},
            'inputs': {'y': ['puppylog/y']},
            'delays': {'y': 4},
            }
        }),
    # do some plotting
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                'd3': ['puppyslice/x_gyr'],
                'd4': ['accint/Ix_acc'], # 'puppylog/y']
                'd5': ['motordel/dy'], # 'puppylog/y']
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd3', 'plot': timeseries},
                    {'input': 'd3', 'plot': histogram},
                ],
                [
                    {'input': 'd4', 'plot': timeseries},
                    {'input': 'd4', 'plot': histogram},
                ],
                [
                    {'input': 'd5', 'plot': timeseries},
                    {'input': 'd5', 'plot': histogram},
                ],
            ]
        },
    }),
    ('plot2', {
        'block': SnsMatrixPlotBlock2,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {
                'd3': ['puppyslice/x_gyr'],
                # 'd3': ['motordiff/dy'],
                'd4': ['motordel/dy'],
                # 'd4': ['puppylog/y'],
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                [
                    # stack inputs into one vector (stack, combine, concat
                    {'input': ['d3', 'd4'], 'mode': 'stack',
                         'plot': histogramnd},
                ],
            ],
        },
    })
])
