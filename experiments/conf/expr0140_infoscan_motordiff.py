"""sensorimotor manifolds: element-wise / motordiff

Plot the sensorimotor manifold / pointcloud, example data from andi
gerken's puppy

Perform element-wise information scan instead of lumped multivariate.
"""

from smp_base.plot import histogramnd
from smp_graphs.common import escape_backslash
from smp_graphs.utils_conf_meas import make_input_matrix, make_input_matrix_ndim
from smp_graphs.block_plot import SnsMatrixPlotBlock2, ImgPlotBlock2
from smp_graphs.block import dBlock2, IBlock2, SliceBlock2, DelayBlock2
from smp_graphs.block_meas import XCorrBlock2
from smp_graphs.block_meas_infth import JHBlock2, MIBlock2, InfoDistBlock2, TEBlock2, CTEBlock2, MIMVBlock2, TEMVBlock2

# showplot = False
randseed = 19482
saveplot = False

outputs = {
    'latex': {'type': 'latex',},
}

# numsteps = 147000
# numsteps = 10000
numsteps = 2000
xdim = 6
ydim = 4

# numsteps = 1000
# xdim = 1
# ydim = 1

# numsteps = 5000
# xdim = 2
# ydim = 1


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
    'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    # 'numsteps': 1000,
    #'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle', # continuous sweep without battery
    'numsteps': 1000,
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'ydim_eff': 1,
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

lconf = {
    'delay_embed_len': 1,
}
    
cnf = ppycnf2
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if cnf.has_key('sys_slicespec'):
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = 0
scanstop = 20 # -10
scanlen = scanstop - scanstart

delay_embed_len = lconf['delay_embed_len']

# prepare scan plot xticks depending on input size
plot_infoscan_xticks_step = scanlen // 5
plot_infoscan_xticks = range(0, scanlen, plot_infoscan_xticks_step)
plot_infoscan_xticklabels = range(scanstart*1, scanstop*1, plot_infoscan_xticks_step*1)

lrp_alpha = 0.01
tap_thr = 0.3

datasetname = escape_backslash(cnf['logfile'])
data_x = 'puppyzero/x_r'
data_y = 'puppylog/y'

desc = """Complete pairwise infoscan for each motor / sensor pair of
variables."""

# def make_input_matrix(id = 'xcorr', base = 'xcorr', xdim = 1, ydim = 1, with_t = False):
#     import numpy as np
#     global scanstart, scanstop, scanlen
#     d = {'d3_%d_%d' % (i, j): {'bus': '%s/%s_%d_%d' % (id, base, i, j)} for j in range(xdim) for i in range(ydim)}
#     if with_t:
#         d['t'] = {'val': np.linspace(scanstart, scanstop-1, scanlen)}
#     # print d
#     return d

# def make_input_matrix_ndim(id = 'xcorr', base = 'xcorr', xdim = 1, ydim = 1, with_t = False):
#     import numpy as np
#     global scanstart, scanstop, scanlen
#     # d = {'d3_%d_%d' % (i, j): {'bus': '%s/%s_%d_%d' % (id, base, i, j)} for j in range(xdim) for i in range(ydim)}
#     d = {}
#     d['d3'] = {'bus': "%s/%s" % (id, base), 'shape': (ydim, xdim, scanlen)} # 
#     if with_t:
#         d['t'] = {'val': np.linspace(scanstart, scanstop-1, scanlen)}
#     # print d
#     return d

graph = OrderedDict([
    # get the data from logfile
    ('puppylog', {
        'block': FileBlock2,
        'params': {
            'id': 'puppylog',
            'inputs': {},
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
            ],
            'blocksize': numsteps,
            'outputs': {
                'log': {'shape': None},
                'x': {'shape': (xdim, numsteps)}, 'y': {'shape': (ydim, numsteps)}
            },
        }
    }),
        
    # mean removal / mu-sigma-res coding
    ('puppyzero', {
        'block': ModelBlock2,
        'params': {
            'debug': True,
            'blocksize': numsteps,
            'inputs': {
                'x': {'bus': 'puppylog/x', 'shape': (xdim, numsteps)},
            },
            'models': {
                'msr': {'type': 'msr'},
            },
        }
    }),
    
    # # puppy process data block: integrate acc, diff motors
    # ('puppylogdiff', {
    #     'block': dBlock2,
    #     'enable': False,
    #     'params': {
    #         'id': 'puppylogdiff',
    #         'blocksize': numsteps,
    #         'inputs': {'y': {'bus': 'puppylog/y'}},
    #         'outputs': {},
    #         'd': 0.1,
    #         'leak': 0.01,
    #         },
    #     }),
    
    # # cross correlation analysis of data
    # ('xcorr', {
    #     'block': XCorrBlock2,
    #     'enable': False,
    #     'params': {
    #         'id': 'xcorr',
    #         'debug': False,
    #         'blocksize': numsteps,
    #         'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylogdiff/dy'}},
    #         'shift': (scanstart, scanstop),
    #         'outputs': {'xcorr': {'shape': (ydim, xdim, scanlen)}}
    #     }
    # }),
        
    # mutual information analysis of data
    ('mi', {
        'block': LoopBlock2,
        'params': {
            'id': 'mi',
            'loopmode': 'parallel',
            'loop': [
                ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}}),
                # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': MIBlock2,
                'params': {
                    'id': 'mi',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'mi': {'shape': (ydim, xdim, scanlen)}}
                }
            },
        }
    }),
    
    # information distance analysis of data
    ('infodist', {
        'block': LoopBlock2,
        'enable': False,
        'params': {
            'id': 'infodist',
            'loopmode': 'parallel',
            'loop': [('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': InfoDistBlock2,
                'params': {
                    'id': 'infodist',
                    'blocksize': numsteps,
                    'debug': False,
                    'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    # 'outputs': {'infodist': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'infodist': {'shape': (ydim, xdim, scanlen)}}
                }
            },
        }
    }),
    
    # transfer entropy analysis of data
    ('te', {
        'block': LoopBlock2,
        'params': {
            'id': 'te',
            'loopmode': 'parallel',
            'loop': [('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': TEBlock2,
                'params': {
                    'id': 'te',
                    'blocksize': numsteps,
                    # 'debug': True,
                    'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    'outputs': {'te': {'shape': (ydim, xdim, scanlen)}}
                }
            },
        }
    }),
    
    # conditional transfer entropy analysis of data
    ('cte', {
        'block': LoopBlock2,
        'params': {
            'id': 'cte',
            'loopmode': 'parallel',
            'loop': [('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}}),
                     # ('inputs', {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/r'}}),
                     # ('inputs', {'x': {'bus': 'puppylog/y'}, 'y': {'bus': 'puppylog/r'}}),
            ],
            'loopblock': {
                'block': CTEBlock2,
                'params': {
                    'id': 'cte',
                    'blocksize': numsteps,
                    # 'debug': True,
                    'xcond': True,
                    'inputs': {'x': {'bus': data_x}, 'y': {'bus': 'puppylog/y'}, 'cond': {'bus': 'puppylog/y'}},
                    'shift': (scanstart, scanstop),
                    'outputs': {'cte': {'shape': (ydim, xdim, scanlen)}}
                }
            },
        }
    }),
        
    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslice', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'puppyslice',
    #         'blocksize': numsteps,
    #         # puppy sensors
    #         'inputs': {'x': {'bus': data_x}},
    #         'slices': {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}},
    #         }
    #     }),
    
    # # puppy process data block: integrate acc, diff motors
    # ('accint', {
    #     'block': IBlock2,
    #     'params': {
    #         'id': 'accint',
    #         'blocksize': numsteps,
    #         'inputs': {'x_acc': {'bus': 'puppyslice/x_acc'}},
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
            # 'inputs': {'y': {'bus': 'puppylogdiff/dy'}},
            'inputs': {'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)}},
            'delays': {'y': 3},
            }
        }),
    
    # do some plotting
    ('plot', {
        'block': PlotBlock2,
        'params': {
            # 'debug': True,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'blocksize': numsteps,
            'inputs': {
                # 'd3': {'bus': 'puppyslice/x_gyr'},
                # 'd4': {'bus': 'accint/Ix_acc'}, # 'puppylog/y'}
                # 'd5': {'bus': 'motordel/dy'}, # 'puppylog/y'}
                'd3': {'bus': data_x},
                'd4': {'bus': 'puppylog/y'}, # 'puppylog/y'}
                # 'd5': {'bus': 'motordel/dy'}, # 'puppylog/y'}
                # 'd6': {'bus': 'puppylog/y'}, # /t
            },
            'outputs': {},#'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {
                        'input': 'd3', 'plot': timeseries,
                        'ylim': (-1., 1.),
                    },
                    {
                        'input': 'd3', 'plot': histogram,
                        'ylim': (-1., 1.),
                    },
                ],
                [
                    {
                        'input': 'd4', 'plot': timeseries,
                    },
                    {
                        'input': 'd4', 'plot': histogram,
                    },
                ],
                # [
                #     {'input': ['d3', 'd4', 'd5'], 'plot': partial(timeseries, marker = ".")},
                #     {'input': 'd6', 'plot': timeseries},
                # ],
            ]
        },
    }),
    
    # plot xcorr
    ('plot_xcor_line', {
        'block': PlotBlock2,
        'enable': False,
        'params': {
            'id': 'plot_xcor_line',
            'logging': False,
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': make_input_matrix_ndim(
                xdim = xdim, ydim = ydim,
                with_t = True, scan = (scanstart, scanstop)),
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [{'input': ['d3'], 'ndslice': (slice(scanlen), i, j), 'xaxis': 't',
                      'shape': (1, scanlen),
                      'plot': partial(timeseries, linestyle="-", marker=".")} for j in range(xdim)]
            for i in range(ydim)],
        },
    }),

    # plot cross-correlation matrix
    ('plot_xcor_img', {
        'block': ImgPlotBlock2,
        'enable': False,
        'params': {
            'id': 'plot_xcor_img',
            'logging': False,
            'saveplot': saveplot,
            # 'debug': False,
            'blocksize': numsteps,
            'desc':  'Plot xcorr img',
            'title': 'Plot xcorr img',
            # 'inputs': make_input_matrix(xdim = xdim, ydim = ydim, with_t = True),
            'inputs': make_input_matrix_ndim(
                xdim = xdim, ydim = ydim,
                with_t = True, scan = (scanstart, scanstop)),
            # 'outputs': {}, #'x': {'shape': (3, 1)}},
            'wspace': 0.5,
            'hspace': 0.5,
            # with one subplot and reshape
            'subplots': [
                [
                    {
                        'input': ['d3'],
                        'ndslice': (slice(None), slice(None), slice(None)),
                        'shape': (scanlen, ydim * xdim), 'cmap': 'RdGy',
                        'vmin': -1.0, 'vmax': 1.0,
                    }
                ],

                [
                    {
                        'input': ['d3'],
                        'ndslice': (slice(None), slice(None), slice(None)),
                        'shape': (scanlen, ydim, xdim), 'cmap': 'RdGy',
                        'dimstack': {'x': [1, 2], 'y': [0]},
                        'vmin': -1.0, 'vmax': 1.0,
                    }
                ],
                
            ],
        },
    }),
    
    # plot mi matrix as image
    ('plot_infoscan', {
        'block': ImgPlotBlock2,
        'params': {
            # 'logging': False,
            # 'debug': False,
            'wspace': 0.2,
            'hspace': 0.2,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'savesize': (4 * 3, 1.5 * 3),
            # 'inputs': {'d1': {'bus': 'mi_1/mi'}, 'd2': {'bus': 'infodist_1/infodist'}},
            'inputs': {
                'd1': {'bus': 'mi_ll0_ll0/mi', 'shape': (ydim, xdim, scanlen)},
                # 'd11': {'bus': 'infodist_ll0_ll0/infodist', 'shape': (ydim, xdim, scanlen)},
                'd2': {'bus': 'te_ll0_ll0/te', 'shape': (ydim, xdim, scanlen)},
                'd3': {'bus': 'cte_ll0_ll0/cte', 'shape': (ydim, xdim, scanlen)}
            },
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {
                    'input': ['d1'],
                    'title': 'MI',
                    'ndslice': (slice(None), slice(None), slice(None)),
                    'shape': (scanlen, ydim, xdim),
                    'dimstack': {'x': [0], 'y': [2, 1]},
                    'cmap': 'Reds',
                    'xlabel': 'Lag [n]',
                    'ylabel': 'sensor channel [i]',
                    'yticks': np.arange(xdim) * ydim + ydim/2,
                    'yticklabels': ['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z'],
                    },
                    
                    # {
                    # 'input': ['d11'],
                    # 'title': 'Infodist',
                    # 'ndslice': (slice(None), slice(None), slice(None)),
                    # 'shape': (scanlen, ydim, xdim),
                    # 'dimstack': {'x': [0], 'y': [2, 1]},
                    # 'cmap': 'Reds'
                    # },
                    
                    {
                    'input': 'd2',
                    'title': 'TE',
                    'ndslice': (slice(None), slice(None), slice(None)),
                    'shape': (scanlen, ydim, xdim),
                    'dimstack': {'x': [0], 'y': [2, 1]},
                    'cmap': 'Reds',
                        'xlabel': 'Lag [n]',
                    'yticks': np.arange(xdim) * ydim + ydim/2,
                    },
                    
                    {
                    'input': 'd3',
                    'title': 'CTE',
                    'ndslice': (slice(None), slice(None), slice(None)),
                    'dimstack': {'x': [0], 'y': [2, 1]},
                    'shape': (scanlen, ydim, xdim),
                    'cmap': 'Reds',
                        'xlabel': 'Lag [n]',
                    'yticks': np.arange(xdim) * ydim + ydim/2,
                    },
                ]
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
    #             # 'd3': {'bus': 'puppyslice/x_gyr'},
    #             'd3': {'bus': data_x},
    #             # 'd3': {'bus': 'motordel/dx'},
    #             # 'd4': {'bus': 'puppylog/y'},
    #             'd4': {'bus': 'motordel/dy'},
    #         },
    #         'outputs': {},#'x': {'shape': (3, 1)}},
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
