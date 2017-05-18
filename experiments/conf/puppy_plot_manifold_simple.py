"""smp_graphs

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2

# showplot = False

randseed = 12345

ppycnf = {
    'numsteps': 27000,
    'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    # 'numsteps': 10000,
    # 'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'numsteps': 2000,
    # 'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

testcnfsin = {
    'numsteps': 1000,
    'xdim': 1,
    'xdim_eff': 1,
    'ydim': 1,
    'logfile': 'data/experiment_20170512_171352_generate_sin_noise_pd.h5',
    'logtype': 'selflog',
}

sphrcnf = {
	'numsteps': 5000,
	'xdim': 2,
    'xdim_eff': 1,
	'ydim': 1,
    'logtype': 'sphero_res_learner',
    # 'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
    'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
}
    
testcnf = {
    'numsteps': 1000,
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logfile': 'data/testlog3.npz',
    'logtype': 'testdata1',
}

cnf = ppycnf
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']

scanstart = -40
scanstop = 1
scanlen = scanstop - scanstart
    
def make_input_matrix(id = 'xcorr', base = 'xcorr', xdim = 1, ydim = 1, with_t = False):
    import numpy as np
    global scanstart, scanstop, scanlen
    d = {'d3_%d_%d' % (i, j): ['%s/%s_%d_%d' % (id, base, i, j)] for j in range(xdim) for i in range(ydim)}
    if with_t:
        d['t'] = [np.linspace(scanstart, scanstop-1, scanlen)]
    # print d
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
            # 'slices': {'x': {'gyr': slice(0, 1)}},
            }
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
    
    # cross correlation analysis of data
    # do this with loopblock :)
    ('xcorr', {
        'block': XCorrBlock2,
        'params': {
            'id': 'xcorr',
            'blocksize': numsteps,
            'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y']},
            'shift': (scanstart, scanstop),
            'outputs': {'xcorr': [(ydim, xdim, scanlen)]}
            }
        }),

    # joint entropy analysis
    ('jh', {
        'block': JHBlock2,
        'params': {
            'id': 'jh',
            'blocksize': numsteps,
            'debug': False,
            'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
            'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
            'outputs': {'jh': [(1, 1)]}
        }
    }),
        
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
                    'debug': False,
                    'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
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
                    'debug': False,
                    'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
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
    #                 'debug': False,
    #                 'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y']},
    #                 'shift': (-3, 4),
    #                 # 'outputs': {'infodist': [((ydim + xdim)**2, 1)]}
    #                 'outputs': {'infodist': [(7 * ydim * xdim_eff, 1)]}
    #             }
    #         },
    #     }
    # }),
    
    # mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loop': [('inputs', {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']}),
                     # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
                     # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
            ],
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': [((ydim + xdim)**2, 1)]}
                    'outputs': {'mi': [(scanlen * ydim * xdim_eff, 1)]}
                }
            },
        }
    }),
    
    # mutual information analysis of data
    ('te', {
        'block': LoopBlock2,
        'params': {
            'id': 'te',
            'loop': [('inputs', {'x': ['puppyslice/x_gyr'], 'y': ['puppylog/y']}),
                     # ('inputs', {'x': ['puppylog/x'], 'y': ['puppylog/r']}),
                     # ('inputs', {'x': ['puppylog/y'], 'y': ['puppylog/r']}),
            ],
            'loopblock': {
                'block': TEBlock2,
                'params': {
                    'id': 'te',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': ['gyrodel/dy'], 'y': ['puppylog/y']},
                    'shift': (scanstart, scanstop),
                    'outputs': {'te': [(scanlen * ydim * xdim_eff, 1)]}
                }
            },
        }
    }),
    
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
    #                 'debug': False,
    #                 'inputs': {'x': ['puppylog/x'], 'y': ['puppylog/y'], 'cond': ['puppylog/y']},
    #                 'shift': (-10, 11),
    #                 'outputs': {'cte': [(scanlen * ydim * xdim, 1)]}
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
    #         'shift': (scanstart, scanstop),
    #         'outputs': {'mi': [(ydim, xdim_eff, scanlen)]}
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
    ('puppystack', {
        'block': StackBlock2,
        'params': {
            'id': 'puppystack',
            'blocksize': numsteps,
            # puppy sensors
            'inputs': make_input_matrix('xcorr', 'xcorr', xdim = xdim, ydim = ydim),
            # 'inputs': {'x': ['puppylog/y']},
            # 'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
            'outputs': {'y': [(xdim * ydim, 1)]} # overwrite
        }
    }),
    
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslicemd', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'puppyslicemd',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': {'x': ['motordel/dy']},
    #         'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    #     }
    # }),
    
    # # plot raw data timeseries and histograms
    # ('plot', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'id': 'plot',
    #         'logging': False,
    #         'debug': False,
    #         'blocksize': numsteps,
    #         'wspace': 0.5,
    #         'hspace': 0.5,
    #         'saveplot': True,
    #         'inputs': {
    #             'd3': ['puppyslice/x_gyr'],
    #             # 'd4': ['accint/Ix_acc'], # 'puppylog/y']
    #             # 'd3': ['puppylog/x'],
    #             'd4': ['puppylog/y'], # 'puppylog/y']
    #             'd5': ['motordel/dy'], # 'puppylog/y']
    #             'd6': ['puppyslicem/x_y0'], # /t
    #             'd7': ['puppyslicemd/x_y0'], # /t
    #         },
    #         'outputs': {},#'x': [(3, 1)]},
    #         'subplots': [
    #             [
    #                 {'input': ['d3'], 'plot': timeseries},
    #                 {'input': 'd3', 'plot': histogram, 'title': 'Sensor histogram'},
    #             ],
    #             [
    #                 {'input': ['d5'], 'plot': timeseries},
    #                 {'input': 'd5', 'plot': histogram},
    #             ],
    #             [
    #                 # {'input': ['d6', 'd7'], 'plot': partial(timeseries, marker = ".")},
    #                 {'input': ['d3', 'd4'], 'plot': partial(timeseries, marker = ".")},
    #                 {'input': 'd6', 'plot': timeseries},
    #             ],
    #         ]
    #     },
    # }),
    
    # plot cross-correlation matrix
    ('plot_xcor_line', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot_xcor_line',
            'logging': False,
            'debug': False,
            'saveplot': True,
            'blocksize': numsteps,
            'inputs': make_input_matrix(xdim = xdim, ydim = ydim, with_t = True),
            'outputs': {}, #'x': [(3, 1)]},
            'subplots': [
                [{'input': 'd3_%d_%d' % (i, j), 'xslice': (0, scanlen), 'xaxis': 't',
                  'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(xdim)]
            for i in range(ydim)],
        },
    }),

    # plot cross-correlation matrix
    ('plot_xcor_img', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_xcor_img',
            'logging': False,
            'saveplot': True,
            'debug': False,
            'blocksize': numsteps,
            'inputs': make_input_matrix(xdim = xdim, ydim = ydim, with_t = True),
            'outputs': {}, #'x': [(3, 1)]},
            'wspace': 0.5,
            'hspace': 0.5,
            'subplots': [
                [{'input': 'd3_%d_%d' % (i, j), 'xslice': (0, scanlen), 'yslice': (0, 1),
                  'shape': (1, scanlen), 'cmap': 'RdGy', 'title': 'xcorrs',
                              'vmin': -1.0, 'vmax': 1.0, 'vaxis': 'cols',} for j in range(xdim)] # 'seismic'
            for i in range(ydim)],
        },
    }),
    
    # plot multivariate (global) mutual information over timeshifts
    ('plot_jh_mimv_temv', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_jh_mimv_temv',
            'logging': False,
            'saveplot': True,
            'debug': False,
            'blocksize': numsteps,
            'inputs': {'d1': ['mimv_1/mimv'], 'd2': ['temv_1/temv'], 't': [np.linspace(scanstart, scanstop-1, scanlen)],
                       'd3': ['jh/jh']},
            'outputs': {}, #'x': [(3, 1)]},
            'subplots': [
                [
                    {'input': 'd3', 'xslice': (0, scanlen), 'yslice': (0, 1), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    'title': 'Joint entropy H(X_1,...,X_n,Y_1,...,Y_n) for time shifts [0, ..., 20]',
                    'shape': (1, scanlen)},
                    {'input': 'd1', 'xslice': (0, scanlen), 'yslice': (0, 1), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'shape': (1, scanlen)},
                    {'input': 'd2', 'xslice': (0, scanlen), 'yslice': (0, 1), 'xaxis': 't',
                     'plot': partial(timeseries, linestyle="none", marker="o"), 'cmap': 'Reds',
                    'title': 'Multivariate transfer entropy TE(Y;X;X^-) for time shifts [0, ..., 20]',
                    'shape': (1, scanlen)}
                ],
            ]
        },
    }),
    
    # plot mi matrix as image
    ('plot_mi_te', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_mi_te',
            'logging': False,
            'saveplot': True,
            'debug': False,
            'wspace': 0.1,
            'hsapce': 0.1,
            'blocksize': numsteps,
            # 'inputs': {'d1': ['mi_1/mi'], 'd2': ['infodist_1/infodist']},
            'inputs': {'d1': ['mi_1/mi'], 'd2': ['te_1/te']}, #, 'd3': ['cte_1/cte']},
            'outputs': {}, #'x': [(3, 1)]},
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, (xdim + ydim)**2),
            #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, (xdim + ydim)**2),
            #              'shape': (xdim+ydim, xdim+ydim), 'plot': 'bla'},
            #     ],
            # ],
            # 'subplots': [
            #     [
            #         {'input': 'd1', 'xslice': (0, xdim * ydim),
            #              'shape': (ydim, xdim), 'plot': 'bla'},
            #         {'input': 'd2', 'xslice': (0, xdim * ydim),
            #              'shape': (ydim, xdim), 'plot': 'bla'},
            #     ],
            # ],
            'subplots': [
                [
                    {'input': 'd1',
                     'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     'xslice': (0, 1),
                     'shape': (ydim, xdim_eff),
                     'title': 'mi-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'plot': 'bla'} for i in range(scanlen)
                ],
                [
                    {'input': 'd2',
                     'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     'xslice': (0, 1),
                     'title': 'te-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'shape': (ydim, xdim_eff), 'plot': 'bla'} for i in range(scanlen)
                ],
                # [
                #     {'input': 'd3', 'xslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                #          'shape': (ydim, xdim_eff), 'plot': 'bla'} for i in range(scanlen)
                # ]
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
