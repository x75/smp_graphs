"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2

# showplot = False

randseed = 12345

ppycnf = {
    # 'numsteps': 147000,
    # 'numsteps': 10000,
    'numsteps': 2000,
    'xdim': 6,
    'ydim': 4,
    'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    # 'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    'logtype': 'selflog',
}

testcnfsin = {
    'numsteps': 1000,
    'xdim': 1,
    'ydim': 1,
    'logfile': 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
    'logtype': 'selflog',
}

sphrcnf = {
	'numsteps': 5000,
	'xdim': 2,
	'ydim': 1,
    'logtype': 'sphero_res_learner',
    # 'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
    'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
}
    
testcnf = {
    'numsteps': 1000,
    'xdim': 6,
    'ydim': 4,
    'logfile': 'data/testlog3.npz',
    'logtype': 'testdata1',
}

cnf = ppycnf
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']

    
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
            # 'type': 'selflog',
            # 'type': 'sphero_res_learner',
            # 'type': 'testdata1',
            'type': cnf['logtype'],
            'file': [
                cnf['logfile'],
                # all files 147000
                # 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5',
                # medium version 10000
                # 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5',
                # short version 2000
                # 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5',
                # test data
                # 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
                # 'data/experiment_20170512_170835_generate_sin_noise_pd.h5',
                # 'data/experiment_20170512_153409_generate_sin_noise_pd.h5',
                # '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
                # '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
                # 'data/testlog3.npz',
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
            'slices': {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}},
            # 'slices': {'x': {'gyr': slice(0, xdim)}},
            }
        }),
    
    # puppy process data block: delay motors by lag to align with their sensory effects
    ('gyrodel', {
        'block': DelayBlock2,
        'params': {
            'id': 'gyrodel',
            'blocksize': numsteps,
            # 'inputs': {'y': ['motordiff/dy']},
            'inputs': {'y': ['puppyslice/x_gyr']},
            'delays': {'y': 1},
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

    # joint entropy analysis
    ('jh', {
        'block': JHBlock2,
        'params': {
            'id': 'jh',
            'blocksize': numsteps,
            'debug': True,
            'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
            'shift': (-20, 1),
                    # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
            'outputs': {'jh': [(1, 1)]}
        }
    }),
        
    # # mutual information analysis of data
    # ('mi', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'mi',
    #         'loop': [('inputs', {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']}),
    #                  # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
    #                  # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
    #         ],
    #         'loopblock': {
    #             'block': MIBlock2,
    #             'params': {
    #                 'id': 'mi',
    #                 'blocksize': numsteps,
    #                 'debug': True,
    #                 'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
    #                 # 'shift': (-120, 8),
    #                 'shift': (-10, 11),
    #                 # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
    #                 'outputs': {'mi': [(21 * ydim * (xdim - 3), 1)]}
    #             }
    #         },
    #     }
    # }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('mimv', {
        'block': LoopBlock2,
        'params': {
            'id': 'mimv',
            'loop': [('inputs', {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']}),
                     # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
                     # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
            ],
            'loopblock': {
                'block': MIMVBlock2,
                'params': {
                    'id': 'mimv',
                    'blocksize': numsteps,
                    'debug': True,
                    'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
                    'shift': (-20, 1), # len 21
                    # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
                    'outputs': {'mimv': [(1, 1)]}
                }
            },
        }
    }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('temv', {
        'block': LoopBlock2,
        'params': {
            'id': 'temv',
            'loop': [('inputs', {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']}),
                     # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
                     # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
            ],
            'loopblock': {
                'block': TEMVBlock2,
                'params': {
                    'id': 'temv',
                    'blocksize': numsteps,
                    'debug': True,
                    'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
                    'shift': (-20, 1), # len 21
                    # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
                    'outputs': {'temv': [(1, 1)]}
                }
            },
        }
    }),
    
    # # mutual information analysis of data
    # ('infodist', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'infodist',
    #         'loop': [('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/y']}),
    #                  # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
    #                  # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
    #         ],
    #         'loopblock': {
    #             'block': InfoDistBlock2,
    #             'params': {
    #                 'id': 'infodist',
    #                 'blocksize': numsteps,
    #                 'debug': True,
    #                 'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y']},
    #                 'shift': (-3, 4),
    #                 # 'outputs': {'infodist': [((ydim + xdim)**2, 1)]}
    #                 'outputs': {'infodist': [(7 * ydim * (xdim - 3), 1)]}
    #             }
    #         },
    #     }
    # }),
    
    # # mutual information analysis of data
    # ('te', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'te',
    #         'loop': [('inputs', {'x': ['gyrodel/dy'], 'y': ['puppylog/y']}),
    #                  # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
    #                  # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
    #         ],
    #         'loopblock': {
    #             'block': TEBlock2,
    #             'params': {
    #                 'id': 'te',
    #                 'blocksize': numsteps,
    #                 'debug': True,
    #                 'inputs': {'x': ['gyrodel/dy'], 'y': ['puppylog/y']},
    #                 'shift': (-10, 11),
    #                 'outputs': {'te': [(21 * ydim * (xdim - 3), 1)]}
    #             }
    #         },
    #     }
    # }),
    
    # # mutual information analysis of data
    # ('cte', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'cte',
    #         'loop': [('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/y'], 'cond': ['puppylog/y']}),
    #                  # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
    #                  # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
    #         ],
    #         'loopblock': {
    #             'block': CTEBlock2,
    #             'params': {
    #                 'id': 'cte',
    #                 'blocksize': numsteps,
    #                 'debug': True,
    #                 'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y'], 'cond': ['puppylog/y']},
    #                 'shift': (-10, 11),
    #                 'outputs': {'cte': [(21 * ydim * xdim, 1)]}
    #             }
    #         },
    #     }
    # }),
    
    # # mutual information analysis of data
    # ('mi2', {
    #     'block': MIBlock2,
    #     'params': {
    #         'id': 'mi2',
    #         'blocksize': numsteps,
    #         'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/r']},
    #         'shift': (-20, 21),
    #         'outputs': {'mi': [(ydim, (xdim - 3), 41)]}
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
            'delays': {'y': 4},
            }
        }),
    
    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslicem', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslicem',
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'x': ['puppylog/y']},
            'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
        }
    }),
    
    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslicemd', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslicemd',
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'x': ['motordel/dy']},
            'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
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
            'wspace': 0.2,
            'hspace': 0.2,
            'inputs': {
                'd3': ['puppyslice/x_gyr'],
                # 'd4': ['accint/Ix_acc'], # 'puppylog/y']
                # 'd3': ['puppylog/x'],
                'd4': ['puppylog/y'], # 'puppylog/y']
                'd5': ['motordel/dy'], # 'puppylog/y']
                'd6': ['puppyslicem/x_y0'], # /t
                'd7': ['puppyslicemd/x_y0'], # /t
            },
            'outputs': {},#'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': ['d3'], 'plot': timeseries},
                    {'input': 'd3', 'plot': histogram},
                ],
                [
                    {'input': ['d5'], 'plot': timeseries},
                    {'input': 'd5', 'plot': histogram},
                ],
                [
                    # {'input': ['d6', 'd7'], 'plot': partial(timeseries, marker = ".")},
                    {'input': ['d3', 'd4'], 'plot': partial(timeseries, marker = ".")},
                    {'input': 'd6', 'plot': timeseries},
                ],
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

    # plot multivariate mutual information
    ('plotmimv', {
        'block': PlotBlock2,
        'params': {
            'id': 'plotmimv',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {'d1': ['mimv_1/mimv'], 'd2': ['temv_1/temv'], 't': [np.linspace(-20, 0, 21)],
                       'd3': ['jh/jh']},
            'outputs': {}, #'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd3', 'xslice': (0, 21), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o")}
                ],
                [
                    {'input': 'd1', 'xslice': (0, 21), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o")}
                ],
                [
                    {'input': 'd2', 'xslice': (0, 21), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o")}
                ],
            ]
        },
    }),
    
    # # plot mi matrix as image
    # ('plotim', {
    #     'block': ImgPlotBlock2,
    #     'params': {
    #         'id': 'plotim',
    #         'logging': False,
    #         'debug': False,
    #         'wspace': 0.1,
    #         'hsapce': 0.1,
    #         'blocksize': numsteps,
    #         # 'inputs': {'d1': ['mi_1/mi'], 'd2': ['infodist_1/infodist']},
    #         'inputs': {'d1': ['mi_1/mi'], 'd2': ['te_1/te'], 'd3': ['cte_1/cte']},
    #         'outputs': {}, #'x': [(3, 1)]},
    #         # 'subplots': [
    #         #     [
    #         #         {'input': 'd1', 'xslice': (0, (xdim + ydim)**2),
    #         #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
    #         #         {'input': 'd2', 'xslice': (0, (xdim + ydim)**2),
    #         #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
    #         #     ],
    #         # ],
    #         # 'subplots': [
    #         #     [
    #         #         {'input': 'd1', 'xslice': (0, xdim * ydim),
    #         #              'shape': (ydim, xdim), 'plot': 'bla'},
    #         #         {'input': 'd2', 'xslice': (0, xdim * ydim),
    #         #              'shape': (ydim, xdim), 'plot': 'bla'},
    #         #     ],
    #         # ],
    #         'subplots': [
    #             [
    #                 {'input': 'd1', 'xslice': (i * (xdim - 3) * ydim, (i+1) * (xdim - 3) * ydim),
    #                      'shape': (ydim, (xdim - 3)), 'plot': 'bla'} for i in range(21)
    #             ],
    #             [
    #                 {'input': 'd2', 'xslice': (i * (xdim - 3) * ydim, (i+1) * (xdim - 3) * ydim),
    #                      'shape': (ydim, (xdim - 3)), 'plot': 'bla'} for i in range(21)
    #             ],
    #             # [
    #             #     {'input': 'd3', 'xslice': (i * (xdim - 3) * ydim, (i+1) * (xdim - 3) * ydim),
    #             #          'shape': (ydim, (xdim - 3)), 'plot': 'bla'} for i in range(21)
    #             # ]
    #         ],
    #     },
    # }),
    
    # # sns matrix plot
    # ('plot2', {
    #     'block': SnsMatrixPlotBlock2,
    #     'params': {
    #         'id': 'plot2',
    #         'logging': False,
    #         'debug': False,
    #         'blocksize': numsteps,
    #         'inputs': {
    #             'd3': ['puppyslice/x_gyr'],
    #             # 'd3': ['puppylog/x'],
    #             # 'd3': ['motordel/dx'],
    #             # 'd4': ['puppylog/y'],
    #             'd4': ['motordel/dy'],
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
