"""smp_graphs config

plot the sensorimotor manifold / pointcloud, example data from andi gerken's puppy
"""

from smp_base.plot import histogramnd
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2, StackBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2

# showplot = False

randseed = 12345

saveplot = False

ppycnf = {
    # 'numsteps': 27000,
    # # 'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'logfile': 'data/experiment_20170526_160018_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    'numsteps': 10000,
    'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'numsteps': 2000,
    # 'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

ppycnf2 = {
    # 'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    'logfile': 'data/stepPickles/step_period_26_0.pickle',
    'numsteps': 1000,
    # 'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle',
    # 'numsteps': 5000,
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
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
    'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150315-223835-eta-0.001000-theta-0.200000-g-0.999000-target-sine.npz',
    # 'logfile': '../../smp_infth/sphero_res_learner_1D/log-learner-20150313-224329.npz',
    'sys_slicespec': {'x': {'gyr': slice(0, 1)}},
}
    
testcnf = {
    'numsteps': 1000,
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logfile': 'data/testlog3.npz',
    'logtype': 'testdata1',
}

cnf = ppycnf2
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if 'sys_slicespec' in cnf:
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = -10
scanstop = 0
scanlen = scanstop - scanstart
    
def make_input_matrix(id = 'xcorr', base = 'xcorr', xdim = 1, ydim = 1, with_t = False):
    import numpy as np
    global scanstart, scanstop, scanlen
    d = {'d3_%d_%d' % (i, j): {'bus': '%s/%s_%d_%d' % (id, base, i, j)} for j in range(xdim) for i in range(ydim)}
    if with_t:
        d['t'] = {'val': np.linspace(scanstart, scanstop-1, scanlen)}
    # print d
    return d

def make_input_matrix_ndim(id = 'xcorr', base = 'xcorr', xdim = 1, ydim = 1, with_t = False):
    import numpy as np
    global scanstart, scanstop, scanlen
    # d = {'d3_%d_%d' % (i, j): {'bus': '%s/%s_%d_%d' % (id, base, i, j)} for j in range(xdim) for i in range(ydim)}
    d = {}
    d['d3'] = {'bus': "%s/%s" % (id, base), 'shape': (ydim, xdim, scanlen)} # 
    if with_t:
        d['t'] = {'val': np.linspace(scanstart, scanstop-1, scanlen)}
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
            'blocksize': numsteps,
            'file': {'filename':
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
            },
            'blocksize': numsteps,
            'outputs': {
                'log': {'shape': None},
                'x': {'shape': (xdim, numsteps)}, 'y': {'shape': (ydim, numsteps)}
            }, # , 
        }
    }),

    # ('wav', {
    #     'block': FileBlock2,
    #     'params': {
    #         'blocksize': 1,
    #         'type': 'wav',
    #         # 'file': ['data/res_out.wav'],
    #         'file': {'filename': 'data/res_out.wav', 'filetype': 'wav', 'offset': 100000, 'length': numsteps},
    #         'file': {'filename': '../../smp/sequence/data/blackbird_XC330200/XC330200-1416_299-01hipan.wav', 'filetype': 'wav', 'offset': 0, 'length': numsteps},
            
    #         'outputs': {'x': {'shape': (2, 1)}}
    #         },
    #     }),
    

    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslice',
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'x': {'bus': 'puppylog/x', 'shape': (xdim, numsteps)}},
            'slices': sys_slicespec,
            # 'slices': ,
            }
        }),
            
    # puppy process data block: integrate acc, diff motors
    ('motordiff', {
        'block': dBlock2,
        'params': {
            'id': 'motordiff',
            'blocksize': numsteps,
            'inputs': {'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)}},
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
            'inputs': {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/y'}},
            'shift': (scanstart, scanstop),
            'outputs': {'xcorr': {'shape': (ydim, xdim, scanlen)}},
            }
        }),

    # joint entropy analysis
    ('jh', {
        'block': JHBlock2,
        'params': {
            'id': 'jh',
            'blocksize': numsteps,
            'debug': False,
            'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
            'shift': (scanstart, scanstop),
            # 'outputs': {'mi': [((ydim + xdim)**2, 1)}}
            'outputs': {'jh': {'shape': (1, scanlen)}}
        }
    }),
        
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('mimv', {
        'block': LoopBlock2,
        'params': {
            'id': 'mimv',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIMVBlock2,
                'params': {
                    'id': 'mimv',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'mimv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('temv', {
        'block': LoopBlock2,
        'params': {
            'id': 'temv',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': TEMVBlock2,
                'params': {
                    'id': 'temv',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'temv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),
    
    # # mutual information analysis of data
    # ('infodist', {
    #     'block': LoopBlock2,
    #     'params': {
    #         'id': 'infodist',
    #         'loop': [('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/y'}}),
    #                  # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
    #                  # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
    #         ],
    #         'loopblock': {
    #             'block': InfoDistBlock2,
    #             'params': {
    #                 'id': 'infodist',
    #                 'blocksize': numsteps,
    #                 'debug': False,
    #                 'inputs': {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/y'}},
    #                 'shift': (-3, 4),
    #                 # 'outputs': {'infodist': {'shape': ((ydim + xdim)**2, 1)}}
    #                 'outputs': {'infodist': {'shape': (7 * ydim * xdim_eff, 1)}}
    #             }
    #         },
    #     }
    # }),
    
    # motor/sensor mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    # 'outputs': {'mi': {'shape': (scanlen * ydim * xdim_eff, 1)}}
                    'outputs': {'mi': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # motor/sensor transfer entropy analysis of data
    ('te', {
        'block': LoopBlock2,
        'params': {
            'id': 'te',
            'loopmode': 'parallel',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': TEBlock2,
                'params': {
                    'id': 'te',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    'outputs': {'te': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # motor/sensor cond motor_i!=j conditional transfer entropy analysis of data
    ('cte', {
        'block': LoopBlock2,
        'params': {
            'id': 'cte',
            'loop': [('inputs', {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/x'}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopmode': 'parallel',
            'loopblock': {
                'block': CTEBlock2,
                'params': {
                    'id': 'cte',
                    'blocksize': numsteps,
                    'debug': False,
                    'xcond': True,
                    'inputs': {'x': {'bus': 'puppyslice/x_gyr'}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    # change this to ndim data, dimstack
                    # 'outputs': {'cte': {'shape': (scanlen * ydim * xdim_eff, 1)}}
                    'outputs': {'cte': {'shape': (ydim, xdim_eff, scanlen)}}
                }
            },
        }
    }),
    
    # # # puppy process data block: integrate acc, diff motors
    # # ('accint', {
    # #     'block': IBlock2,
    # #     'params': {
    # #         'id': 'accint',
    # #         'blocksize': numsteps,
    # #         'inputs': {'x_acc': {'bus': 'puppyslice/x_acc'}},
    # #         'outputs': {},
    # #         'd': 0.1,
    # #         'leak': 0.01,
    # #         },
    # #     }),
    
    # puppy process data block: delay motors by lag to align with their sensory effects
    ('motordel', {
        'block': DelayBlock2,
        'params': {
            'id': 'motordel',
            'blocksize': numsteps,
            # 'inputs': {'y': {'bus': 'motordiff/dy'}},
            'inputs': {'y': {'bus': 'puppylog/y'}},
            'delays': {'y': 3},
            }
        }),
    
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslicem', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'puppyslicem',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': {'x': {'bus': 'puppylog/y'}},
    #         'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    #     }
    # }),
     
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppystack', {
    #     'block': StackBlock2,
    #     'params': {
    #         'id': 'puppystack',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': make_input_matrix('xcorr', 'xcorr', xdim = xdim, ydim = ydim),
    #         # 'inputs': {'x': {'bus': 'puppylog/y'}},
    #         # 'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    #         'outputs': {'y': {'shape': (xdim * ydim, 1)}} # overwrite
    #     }
    # }),
    
    # # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # # ('puppyslicemd', {
    # #     'block': SliceBlock2,
    # #     'params': {
    # #         'id': 'puppyslicemd',
    # #         'blocksize': numsteps,
    # #         # puppy sensors
    # #         'inputs': {'x': {'bus': 'motordel/dy'}},
    # #         'slices': {'x': {'y%d' % i: slice(i, i+1) for i in range(ydim)}},
    # #     }
    # # }),
    
    # plot raw data timeseries and histograms
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot',
            'logging': False,
            'debug': False,
            'blocksize': numsteps,
            'wspace': 0.5,
            'hspace': 0.5,
            'saveplot': saveplot,
            'inputs': {
                'd3': {'bus': 'puppyslice/x_gyr'},
                # 'd4': {'bus': 'accint/Ix_acc'}, # 'puppylog/y'}
                # 'd3': {'bus': 'puppylog/x'},
                'd4': {'bus': 'puppylog/y'}, # 'puppylog/y'}
                # 'd5': {'bus': 'motordel/dy'}, # 'puppylog/y'}
                # 'd6': {'bus': 'puppyslicem/x_y0'}, # /t
                # 'd7': {'bus': 'puppyslicemd/x_y0'}, # /t
            },
            'outputs': {},#'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {'input': ['d3'], 'plot': timeseries},
                    {'input': 'd3', 'plot': histogram, 'title': 'Sensor histogram'},
                ],
                [
                    {'input': ['d4'], 'plot': timeseries},
                    {'input': 'd4', 'plot': histogram, 'title': 'Motor histogram'},
                ],
                # [
                #     {'input': ['d5'], 'plot': timeseries},
                #     {'input': 'd5', 'plot': histogram},
                # ],
                # [
                #     # {'input': ['d6', 'd7'], 'plot': partial(timeseries, marker = ".")},
                #     {'input': ['d3', 'd4'], 'plot': partial(timeseries, marker = ".")},
                #     {'input': 'd6', 'plot': timeseries},
                # ],
            ]
        },
    }),
    
    # plot cross-correlation matrix
    ('plot_xcor_line', {
        'block': PlotBlock2,
        'params': {
            'id': 'plot_xcor_line',
            'logging': False,
            'debug': False,
            'saveplot': saveplot,
            'blocksize': numsteps,
            'inputs': make_input_matrix_ndim(xdim = xdim, ydim = ydim, with_t = True),
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            # 'xslice': (0, scanlen), 
            'subplots': [
                [{'input': ['d3'], 'ndslice': (slice(scanlen), i, j), 'xaxis': 't',
                      'shape': (1, scanlen),
                  'plot': partial(timeseries, linestyle="-", marker=".")} for j in range(xdim)]
                for i in range(ydim)],
                
            #     [{'input': 'd3_%d_%d' % (i, j), 'xslice': (0, scanlen), 'xaxis': 't',
            #       'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(xdim)]
            # for i in range(ydim)],
            
        },
    }),

    # plot cross-correlation matrix
    ('plot_xcor_img', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_xcor_img',
            'logging': False,
            'saveplot': saveplot,
            'debug': False,
            'blocksize': numsteps,
            # 'inputs': make_input_matrix(xdim = xdim, ydim = ydim, with_t = True),
            'inputs': make_input_matrix_ndim(xdim = xdim, ydim = ydim, with_t = True),
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            'subplots': [
                # [{'input': ['d3'], 'ndslice': (i, j, ), 'xslice': (0, scanlen), 'xaxis': 't',
                #   'plot': partial(timeseries, linestyle="none", marker=".")} for j in range(xdim)]
                # for i in range(ydim)],
                [{'input': ['d3'], 'ndslice': (slice(scanlen), i, j),
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
            'saveplot': saveplot,
            'debug': False,
            'wspace': 0.5,
            'hspace': 0.5,
            'blocksize': numsteps,
            'inputs': {'d1': {'bus': 'mimv|0/mimv', 'shape': (1, scanlen)},
                       'd2': {'bus': 'temv|0/temv', 'shape': (1, scanlen)},
                       'd3': {'bus': 'jh/jh', 'shape': (1, scanlen)},
                       't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},},
            'outputs': {}, #'x': {'shape': (3, 1)}},
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
            'saveplot': saveplot,
            'debug': False,
            'wspace': 0.1,
            'hsapce': 0.1,
            'blocksize': numsteps,
            # 'inputs': {'d1': {'bus': 'mi_1/mi'}, 'd2': {'bus': 'infodist_1/infodist'}},
            'inputs': {'d1': {'bus': 'mi|0/mi'}, 'd2': {'bus': 'te|0/te'}, 'd3': {'bus': 'cte|0/cte'}},
            'outputs': {}, #'x': {'shape': (3, 1)}},
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
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'shape': (ydim, xdim_eff),
                     'title': 'mi-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'plot': 'bla'} for i in range(scanlen)
                ],
                [
                    {'input': 'd2',
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'title': 'te-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'shape': (ydim, xdim_eff),
                     'plot': 'bla'} for i in range(scanlen)
                ],
                [
                    {'input': 'd3',
                     # 'yslice': (i * xdim_eff * ydim, (i+1) * xdim_eff * ydim),
                     # 'xslice': (0, 1),
                     'ndslice': (i, slice(None), slice(None)),
                     'title': 'cte-matrix', 'cmap': 'Reds',
                     'vaxis': 'rows',
                     'shape': (ydim, xdim_eff),
                     'plot': 'bla'} for i in range(scanlen)
                ],
            ],
        },
    }),
    
    # sns matrix plot
    ('plot2', {
        'block': SnsMatrixPlotBlock2,
        'params': {
            'id': 'plot2',
            'logging': False,
            'debug': False,
            'saveplot': saveplot,
            'blocksize': numsteps,
            'inputs': {
                'd3': {'bus': 'puppyslice/x_gyr'},
                # 'd3': {'bus': 'puppylog/x'},
                # 'd3': {'bus': 'motordel/dx'},
                # 'd4': {'bus': 'puppylog/y'},
                'd4': {'bus': 'motordel/dy'},
            },
            'outputs': {},#'x': {'shape': 3, 1)}},
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
