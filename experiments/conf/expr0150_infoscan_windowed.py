"""smp_graphs

perform windowed short time mutual info scan
"""

from smp_graphs.common import escape_backslash
from smp_graphs.block_meas_infth import MIMVBlock2, CMIMVBlock2, JHBlock2, TEMVBlock2
from smp_graphs.block import SliceBlock2, SeqLoopBlock2, StackBlock2
from smp_graphs.block_plot import ImgPlotBlock2

saveplot = True
recurrent = True
    
outputs = {
    'latex': {'type': 'latex',},
}

lpzbarrelcnf = {
    'numsteps': 2000,
    # 'logfile': 'data/experiment_20170626_120004_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115924_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115813_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115719_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115628_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115540_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115457_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115406_actinf_m1_goal_error_ND_pd.h5', # 250
    # 'logfile': 'data/experiment_20170626_115323_actinf_m1_goal_error_ND_pd.h5', # 250
    # learning static target
    # 'logfile': 'data/experiment_20170626_124247_actinf_m1_goal_error_ND_pd.h5', # 500
    # just stepping
    # 'logfile': 'data/experiment_20170626_125226_actinf_m1_goal_error_ND_pd.h5', # 500
    # 'logfile': 'data/experiment_20170626_140407_actinf_m1_goal_error_ND_pd.h5', # 1000
    'logfile': 'data/smp_dm_actinf_m1_goal_error_ND_embedding/smp_dm_actinf_m1_goal_error_ND_embedding_ffc25da6e3ef540e66b0c98e2642752a_20180103_165831_log_pd.h5',
    # 'logfile': 'data/smp_dm_actinf_m1_goal_error_ND_embedding/smp_dm_actinf_m1_goal_error_ND_embedding_562ac39aaddd5cbabb3fc9f512176b78_20180105_154421_log_pd.h5',
    'xdim': 2,
    'xdim_eff': 2,
    'ydim': 2,
    'logtype': 'selflog',
    'sys_slicespec': {'x': {'gyr': slice(0, 2)}},
    # data_x_key = 's_proprio',
    # data_y_key = 'pre',
    'data_x_key': 's0',
    'data_y_key': 'pre',
}
    
ppycnf = {
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170526_160018_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    # 'numsteps': 10000,
    # 'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    'numsteps': 2000,
    'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', # 2K
    # 'numsteps': 20000,
    # 'logfile': 'data/experiment_20170530_174612_process_logfiles_pd.h5', # step fast-to-slow newB all concatenated
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
    'data_x_key': 'x',
    'data_y_key': 'y',
}

ppycnf2 = {
    # 'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle', # continuous sweep without battery
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 6, # 3,
    'ydim': 4,
    'numsteps': 5000,
    'data_x_key': 'x',
    'data_y_key': 'y',
}

cnf = ppycnf2
# cnf = lpzbarrelcnf

# copy params to namespace
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if cnf.has_key('sys_slicespec'):
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = 0
scanstop = 80 # -30
scanlen = scanstop - scanstart
delay_embed_len = 1

# 1000/1000
# winsize = 2000
# overlap = 2000
# winsize = 1000
# overlap = 1000
winsize = 500
overlap = 500
# winsize = 50
# overlap = 50
srcsize = overlap
# numwins = numsteps/overlap
numwins = (numsteps - winsize)/overlap + 1

# prepare scan plot xticks depending on input size
plot_infoscan_xticks_step = scanlen // 5
plot_infoscan_xticks = range(0, scanlen, plot_infoscan_xticks_step)
plot_infoscan_xticklabels = range(scanstart*1, scanstop*1, plot_infoscan_xticks_step*1)

plot_infoscan_yticks_step = overlap
plot_infoscan_yticks = np.array(range(0, numwins)) + 0.5
plot_infoscan_yticklabels = range(0, numsteps, plot_infoscan_yticks_step)

# print "numwins", numwins, "yticks", plot_infoscan_yticks

data_x = 'puppyzero/x_r'
data_y = 'puppyzero/y_r'
data_x_key = cnf['data_x_key']
data_y_key = cnf['data_y_key']
datasetname = escape_backslash(cnf['logfile'])

desc = """This experiment consists of an open-loop exploration dataset
of {0} time steps. Information scans are applied repeatedly on a
window sliding over the dataset with a step size equal to the window
size. The experiment is meant to highlight the fact that the
information propagation delays in moderately complex robot bodies,
like Puppy, can be time dependent. In the experiment, the motor
frequency sweep ties together time and frequency and each window's
measurement is in direct correspondence with the frequency range swept
within its window. The experiment does not need to account for
hysteresis effects because such effects would on average only increase
the effect. This is in support of the hypothesis, that an agent's
predictive models can gain benefit not only from precise time-modality
tappings but also from a temporal hierarchy of dependency feature
detectors. The episode is explained in detail in the figure
cpation.""".format(numsteps)

loopblocksize = numsteps

loopblock = {
    'block': Block2,
    'params': {
        'id': 'bhier',
        'debug': False,
        'topblock': False,
        'logging': False,
        'numsteps': numsteps,
        'blocksize': 1,
        'blockphase': [0],
        # 'outputs': {'jh': {'shape': (1,1)}},
        'outputs': {'jh': {'shape': (1, 1), 'buscopy': 'jh/jh'}},
        # contains the subgraph specified in this config file
        'graph': OrderedDict([
            ('puppylog', {
                'block': FileBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    'type': cnf['logtype'],
                    'file': {'filename': cnf['logfile']},
                    # 'storekeys': ['/robot1/s0', '/robot1/s1'],
                    'outputs': {
                        'log': None,
                        data_x_key: {'shape': (xdim, numsteps), 'storekey': '/robot1/s0'},
                        data_y_key: {'shape': (ydim, numsteps), 'storekey': '/pre_l0/pre'}
                    },
                },
            }),
                
            # mean removal / mu-sigma-res coding
            ('puppyzero', {
                'block': ModelBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    'inputs': {
                        'x': {'bus': 'puppylog/x', 'shape': (xdim, numsteps)},
                        'y': {'bus': 'puppylog/y', 'shape': (ydim, numsteps)},
                    },
                    'models': {
                        'msr': {'type': 'msr'},
                    },
                }
            }),
        
            # puppy process data block: delay motors by lag to align with their sensory effects
            ('motordel', {
                'block': DelayBlock2,
                'params': {
                    # 'debug': True,
                    'blocksize': numsteps,
                    'flat2': True,
                    # 'inputs': {'y': {'bus': 'motordiff/dy'}},
                    'inputs': {
                        'x': {'bus': data_x, 'shape': (xdim, numsteps)},
                        'y': {'bus': data_y, 'shape': (ydim, numsteps)}
                    },
                    'delays': {
                        'x': range(1, delay_embed_len+1), # [1],
                        # 'y': range(1, delay_embed_len+1), # [1, 0, -1, -2, -3], # * delay_embed_len, # [1],
                        'y': range(0, delay_embed_len), # [1, 0, -1, -2, -3], # * delay_embed_len, # [1],
                    },
                }
            }),

            # joint entropy
            ('jh', {
                'block': JHBlock2,
                'params': {
                    'id': 'jh',
                    'blocksize': numsteps,
                    'debug': False,
                    'logging': False,
                    'inputs': {
                        'x': {'bus': data_x},
                        'y': {'bus': data_y}
                    },
                    'shift': (0, 1),
                    'outputs': {'jh': {'shape': (1, 1)}}
                }
            })
        ]),
    }
}

graph = OrderedDict([
    # a loop block calling the enclosed block len(loop) times,
    # returning data of looplength in one outer step
    ("jhloop", {
        'debug': True,
        'block': SeqLoopBlock2,
        'params': {
            'id': 'jhloop',
            # loop specification, check hierarchical block to completely pass on the contained in/out space?
            'blocksize': numsteps, # same as loop length
            'blockphase': [1],
            'numsteps':  numsteps,
            'loopblocksize': loopblocksize,
            # can't do this dynamically yet without changing init passes
            'outputs': {
                'jh': {'shape': (1, 1)},
            },
            # 'loop': [('inputs', {
            #     'lo': {'val': np.random.uniform(-i, 0, (3, 1)), 'shape': (3, 1)}, 'hi': {'val': np.random.uniform(0.1, i, (3, 1)), 'shape': (3, 1)}}) for i in range(1, 11)],
            # 'loop': lambda ref, i: ('inputs', {'lo': [10 * i], 'hi': [20*i]}),
            # 'loop': [('inputs', {'x': {'val': np.random.uniform(np.pi/2, 3*np.pi/2, (3,1))]}) for i in range(1, numsteps+1)],
            # 'loop': partial(f_loop_hpo, space = f_loop_hpo_space_f3(pdim = 3)),
            'loop': [('none', {})], # lambda ref, i, obj: ('none', {}),
            'loopmode': 'sequential',
            'loopblock': loopblock,
        },
    }),
    
    # ('data', {
    #     'block': FileBlock2,
    #     'params': {
    #         'id': 'data',
    #         'debug': False,
    #         # 'blocksize': overlap, # numsteps,
    #         'blocksize': srcsize, # numsteps,
    #         'type': cnf['logtype'],
    #         'file': {'filename': cnf['logfile']},
    #         'outputs': {'log': None, 'x': {'shape': (xdim, srcsize)},
    #                         'y': {'shape': (ydim, srcsize)}},
    #     },
    # }),

    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('dataslice', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'id': 'dataslice',
    #         # 'blocksize': overlap,
    #         'blocksize': srcsize,
    #         'debug': True,
    #         # puppy sensors
    #         'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, srcsize)}},
    #         'slices': sys_slicespec,
    #         # 'slices': ,
    #         }
    #     }),
        
    # # multivariate mutual information analysis of data I(X^n ; Y^m)
    # ('mimvl', {
    #     'block': LoopBlock2,
    #     'enabled': False,
    #     'params': {
    #         'id': 'mimvl',
    #         'blocksize': overlap,
    #         'debug': False,
    #         'loop': [
    #             ('inputs', {
    #                 'x': {'bus': 'puppyslice/x_acc', 'shape': (xdim_eff, winsize)},
    #                 'y': {'bus': 'puppylog/%s' % data_y_key, 'shape': (ydim, winsize)},
    #                 'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
    #                 # 'norm': {'val': np.array([[7.0]]), 'shape': (1, 1)},
    #             }),
    #             # ('inputs', {'x': {'bus': 'dataslice/x_gyr'}, 'y': {'bus': 'data/y'}}),
    #             # ('inputs', {'x': {'bus': 'data/x'}, 'y': {'bus': 'data/r'}}),
    #             # ('inputs', {'x': {'bus': 'data/y'}, 'y': {'bus': 'data/r'}}),
    #         ],
    #         'loopmode': 'parallel',
    #         'loopblock': {
    #             'block': MIMVBlock2,
    #             'params': {
    #                 'id': 'mimv',
    #                 'blocksize': overlap,
    #                 'debug': False,
    #                 'inputs': {'x': {'bus': 'puppyslice/x_gyr',
    #                                      'shape': (xdim_eff, winsize)},
    #                                'y': {'bus': 'puppylog/%s' % data_y_key,
    #                                          'shape': (ydim, winsize)}},
    #                 # 'shift': (-120, 8),
    #                 'shift': (scanstart, scanstop), # len 21
    #                 # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
    #                 'outputs': {'mimv': {'shape': (1, scanlen)}}
    #             }
    #         },
    #     }
    # }),

    # # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    # ('puppyslice', {
    #     'block': SliceBlock2,
    #     'params': {
    #         'blocksize': srcsize,
    #         'inputs': {'x': {'bus': 'puppylog/%s' % data_x_key, 'shape': (xdim, numsteps)}},
    #         'slices': sys_slicespec,
    #     }
    # }),
                
    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('puppyslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'puppyslice',
            'blocksize': numsteps,
            # puppy sensors
            # 'inputs': {'x': {'bus': data_x, 'shape': (xdim, numsteps)}},
            'inputs': {'x': {'bus': data_x, 'shape': (xdim, numsteps)}},
            'slices': sys_slicespec,
        }
    }),
            
    # slice block to split puppy motors y into single channels
    ('puppyslice_y', {
        'block': SliceBlock2,
        'params': {
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {'y': {'bus': data_y, 'shape': (ydim, numsteps)}},
            'slices': {'y': dict([('c%d' % i, [i]) for i in range(ydim)])},
        }
    }),
                
    # puppy process data block: integrate acc, diff motors
    ('motordiff', {
        'block': dBlock2,
        'params': {
            'id': 'motordiff',
            'blocksize': numsteps,
            'inputs': {'y': {'bus': data_y, 'shape': (ydim, numsteps)}},
            'outputs': {},
            'd': 0.1,
            'leak': 0.01,
        },
    }),
    
    # stack delay x and y together as condition for m -> s mi
    ('motorstack', {
        'block': StackBlock2,
        'params': {
            'blocksize': numsteps,
            # puppy sensors
            'inputs': {
                'x': {'bus': 'motordel/dx', 'shape': (xdim, numsteps)},
                'y': {'bus': 'motordel/dy', 'shape': (ydim, numsteps)},
            },
            'outputs': {'y': {'shape': (
                xdim * delay_embed_len + ydim * delay_embed_len,
                numsteps
            )}},
        }
    }),
    
    # processing
    ('mimv', {
        'block': MIMVBlock2,
        'params': {
            'id': 'mimv',
            'blocksize': overlap,
            'debug': False,
            'inputs': {
                'x': {'bus': data_x, 'shape': (xdim_eff, winsize)},
                'y': {'bus': data_y, 'shape': (ydim, winsize)},
                'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
            },
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
            'outputs': {'mimv': {'shape': (1, scanlen)}}
        }
    }),
    
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('cmimv', {
        'block': CMIMVBlock2,
        'params': {
            'blocksize': overlap,
            # 'debug': True,
            # 'norm_out': False,
            'inputs': {
                'x': {'bus': data_x, 'shape': (xdim_eff, winsize)},
                'y': {'bus': data_y, 'shape': (ydim, winsize)},
                # 'x': {'bus': data_x},
                # # 'y': {'bus': data_y},
                # 'y': {'bus': 'motordel/dy'},
                # 'cond': {'bus': 'motorstack/y'},
                # 'cond': {'bus': 'motordel/dy'},
                'cond_delay': {'bus': 'motordel/dx', 'shape': (xdim_eff, winsize)},
            },
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
            'outputs': {'cmimv': {'shape': (1, scanlen)}}
        },
    }),
    
    ('temv', {
        'block': TEMVBlock2,
        'params': {
            'id': 'temv',
            'blocksize': overlap,
            'debug': False,
            'inputs': {
                'x': {'bus': data_x, 'shape': (xdim_eff, winsize)},
                'y': {'bus': data_y, 'shape': (ydim, winsize)},
                'norm': {'bus': 'jhloop/jh', 'shape': (1, 1)},
            },
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            'outputs': {'temv': {'shape': (1, scanlen)},}
        }
    }),

    # plot sensorimotor timeseries
    ('plot', {
        'block': PlotBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.2, 'hspace': 0.2,
            'title_pos': 'top_out',
            'desc': """Timeseries plot of motor (bottom) and sensor
                (top) measurements during an episode of open-loop,
                frequency-swept sinusoid motor exploration. The episode
                duration in time steps is {0} time steps. The sweep is
                linear in frequency with ($f_{{\text{{start}}}} = 0 Hz,
                f_{{\text{{stop}}}} = 6.4 Hz$)""".format(numsteps, ),
            'inputs': {
                'd1': {'bus': data_x, 'shape': (xdim, numsteps)},
                'd2': {'bus': data_y, 'shape': (ydim, numsteps)}
            },
            'subplots': [
                
                [
                    {
                        'input': 'd1', 'plot': timeseries,
                        'title': 'Sensors',
                        'ylim': (-1., 1.),
                    },
                    
                    {
                        'input': 'd1', 'plot': histogram,
                        'title': 'Sensors',
                        'ylim': (-1., 1.),
                    },
                ],
                
                [
                    {
                        'input': 'd2', 'plot': timeseries,
                        'title': 'Motors',
                    },
                    {
                        'input': 'd2', 'plot': histogram,
                        'title': 'Motors',
                    },
                ],
            ]
        }
    }),
        
    # plot infoscan (multivariate (global) mutual information over timeshifts)
    ('plot_infoscan', {
        'block': ImgPlotBlock2,
        'params': {
            # 'debug': True,
            'saveplot': saveplot,
            'savesize': (4 * 3, 1.5 * 3),
            'savetype': 'pdf',
            'wspace': 0.2,
            'hspace': 0.2,
            'blocksize': overlap, # numsteps,
            'desc': """Multiple windowed info scans for dataset
            {0}. The measures used in the scan from left to right are
            mutual information (MI), transfer entropy (TE) and
            conditional transfer entropy (CTE). The condition for the
            CTE is the motor past. The mutual information cannot
            distinguish between the action's effect and the body's
            dynamic memory, or external entropy. Nonetheless the MI
            seems to be consistent with the more localized
            measurements TE and CTE via an integrating relation. This
            suggests differentiated versions of a conditional mutual
            information CMI with conditions
            $\\textnormal{{sensor}}^{{-}}$ and $\\text{{motor}}^{{-}}$
            with $\cdot^{{-}}$ meaning
            past-of-variable.""".format(datasetname),
            'title': 'Multiple windowed info scans for dataset %s' % (cnf['logfile']),
            'title_pos': 'top_in',
            'inputs': {
                'd1': {'bus': 'mimv/mimv', 'shape': (1, scanlen * numwins)},
                # 'd2': {'bus': 'mimvl|0/mimv', 'shape': (1, scanlen * numwins)},
                'd2': {'bus': 'cmimv/cmimv', 'shape': (1, scanlen * numwins)},
                'd3': {'bus': 'temv/temv', 'shape': (1, scanlen * numwins)},                
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},},
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    
                    {
                        'input': 'd1', 
                        'cmap': 'Reds',
                        'title': 'Mutual information',
                        'ndslice': (slice(None), slice(None)),
                        # 'dimstack': {'x': [0], 'y': [1]},
                        'shape': (numwins, scanlen),
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'xlabel': 'Lag [n]',
                        'ylabel': 'Time [n]',
                        'yticks': plot_infoscan_yticks,
                        'yticklabels': plot_infoscan_yticklabels,
                        'vmin': 0.0,
                    },
                    
                    {
                        'input': 'd2', 
                        'cmap': 'Reds',
                        'title': 'Cond. mutual information ',
                        'ndslice': (slice(None), slice(None)),
                        # 'dimstack': {'x': [0], 'y': [1]},
                        'shape': (numwins, scanlen),
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'xlabel': 'Lag [n]',
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0.0,
                    },
                    
                    {
                        'input': 'd3',
                        'cmap': 'Reds',
                        'title': 'Transfer entropy',
                        'ndslice': (slice(None), slice(None)),
                        # 'dimstack': {'x': [2, 1], 'y': [0]},
                        'shape': (numwins, scanlen),
                        'colorbar': True, 'colorbar_orientation': 'vertical',
                        'xlabel': 'Lag [n]',
                        'ylabel': False,
                        'yticks': False,
                        'vmin': 0.0,
                    },
                    
                ],
            ]
        },
    }),
    
    
])
