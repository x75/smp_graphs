"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import MIBlock2

# numsteps = 147000
numsteps = 10000
# numsteps = 2000
xdim = 6
ydim = 4

# numsteps = 1000
# xdim = 1
# ydim = 1

def make_input_matrix(xdim = 1, ydim = 1):
    import numpy as np
    d = {'d3_%d_%d' % (i, j): ['xcorr/xcorr_%d_%d' % (i, j)] for j in range(xdim) for i in range(ydim)}
    d['t'] = [np.linspace(-20, 20, 41)]
    print d
    return d

graph = OrderedDict([
    # get the data from logfile
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
                'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5',
                # short version 2000
                # 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5',
                # test data
                # 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
                # 'data/experiment_20170512_170835_generate_sin_noise_pd.h5',
                # 'data/experiment_20170512_153409_generate_sin_noise_pd.h5',
                ],
            'blocksize': numsteps,
            'outputs': {'log': [None]},
            }
        }),
        
    # cross correlation analysis of data
    # do this with loopblock :)
    ('xcorr', {
        'block': XCorrBlock2,
        'params': {
            'id': 'xcorr',
            'blocksize': numsteps,
            'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y']},
            'shift': (-20, 21),
            'outputs': {'xcorr': [(ydim, xdim, 41)]}
            }
        }),
        
    # mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loop': [('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/y']}),
                     # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
                     # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
            ],
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': True,
                    'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y']},
                    'shift': (-20, 21),
                    'outputs': {'mi': [(ydim * xdim, 1)]}
                }
            },
        }
    }),
    
    # # mutual information analysis of data
    # ('mi2', {
    #     'block': MIBlock2,
    #     'params': {
    #         'id': 'mi2',
    #         'blocksize': numsteps,
    #         'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/r']},
    #         'shift': (-20, 21),
    #         'outputs': {'mi': [(ydim, xdim, 41)]}
    #         }
    #     }),
    
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslice', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'puppyslice',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': {'x': ['puppylog/x']},
    #         'slices': {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}},
    #         }
    #     }),
    
    # # puppy process data block: integrate acc, diff motors
    # ('accint', {
    #     'block': IBlock2,
    #     'params': {
    #         'id': 'accint',
    #         'blocksize': numsteps,
    #         'inputs': {'x_acc': ['puppyslice/x_acc']},
    #         'outputs': {},
    #         'd': 0.1,
    #         'leak': 0.01,
    #         },
    #     }),
    
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
            'delays': {'y': 1},
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
                # 'd3': ['puppyslice/x_gyr'],
                # 'd4': ['accint/Ix_acc'], # 'puppylog/y']
                # 'd5': ['motordel/dy'], # 'puppylog/y']
                'd3': ['puppylog/x'],
                'd4': ['puppylog/y'], # 'puppylog/y']
                # 'd5': ['motordel/dy'], # 'puppylog/y']
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
                # [
                #     {'input': 'd5', 'plot': timeseries},
                #     {'input': 'd5', 'plot': histogram},
                # ],
            ]
        },
    }),
    
    # plot xcorr
    ('plot3', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot3',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': make_input_matrix(xdim = xdim, ydim = ydim),
            'outputs': {}, #'x': [(3, 1)]},
            'subplots': [
                [{'input': 'd3_%d_%d' % (i, j), 'xslice': (0, 41), 'xaxis': 't',
                  'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(xdim)]
            for i in range(ydim)],
        },
    }),

    # plot mi matrix as image
    ('plotim', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plotim',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {'d1': ['mi_1/mi']},
            'outputs': {}, #'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd1', 'xslice': (0, xdim * ydim), 'plot': 'bla'},
                ],
            ],
        },
    }),
    
    # # sns matrix plot
    # ('plot2', {
    #     'block': SnsMatrixPlotBlock2,
    #     'params': {
    #         'id': 'plot2',
    #         'logging': False,
    #         'debug': False,
    #         'blocksize': numsteps,
    #         'inputs': {
    #             # 'd3': ['puppyslice/x_gyr'],
    #             'd3': ['puppylog/x'],
    #             # 'd4': ['motordel/dy'],
    #             'd4': ['puppylog/y'],
    #         },
    #         'outputs': {},#'x': [(3, 1)]},
    #         'subplots': [
    #             [
    #                 # stack inputs into one vector (stack, combine, concat
    #                 {'input': ['d3', 'd4'], 'mode': 'stack',
    #                      'plot': histogramnd},
    #             ],
    #         ],
    #     },
    # })
])
